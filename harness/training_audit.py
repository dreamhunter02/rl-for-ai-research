"""Synchronous workshop hooks: exclude zero-signal groups and count actual updates."""
import inspect
import json
import time
from pathlib import Path


def install_training_audit(train_module, output_path):
    original = train_module.train_step
    signature = inspect.signature(original)
    counter = {'optimizer_updates': 0, 'skipped_batches': 0, 'calls': 0}
    def record(updates, datums, elapsed, metrics):
        counter['calls'] += 1
        counter['optimizer_updates'] += updates
        counter['skipped_batches'] += int(not updates)
        metrics.update({f'workshop/{k}': v for k, v in counter.items()})
        with Path(output_path).open('a') as f:
            f.write(json.dumps({**counter, 'actual_updates_this_call': updates, 'training_datums': datums,
                                'elapsed_s': elapsed, 'metrics': metrics})+'\n')

    async def audited(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        data = bound.arguments['data_D']
        started = time.monotonic()
        result = await original(*args, **kwargs) if data else []
        updates = min(bound.arguments.get('num_substeps', 1), len(data))
        metrics = bound.arguments.get('metrics')
        record(updates,len(data),time.monotonic()-started, metrics if metrics is not None else {})
        return result
    train_module.train_step = audited

    if hasattr(train_module, 'do_train_step_and_get_sampling_client'):
        # Cookbook intentionally keeps a constant group when all groups are equal.
        # Defer filtering so builder/group pairs remain aligned, then skip the optimizer
        # entirely when no nonzero centered advantage with generated tokens remains.
        train_module.remove_constant_reward_groups = lambda groups: groups
        group_step = train_module.do_train_step_and_get_sampling_client
        group_signature = inspect.signature(group_step)
        async def filtered_step(*args, **kwargs):
            bound = group_signature.bind(*args, **kwargs)
            groups = bound.arguments['trajectory_groups_P']
            builders = bound.arguments['env_group_builders_P']
            if len(groups) != len(builders): raise ValueError('Builder/group alignment lost')
            retained=[]
            for builder,group in zip(builders,groups):
                rewards=group.get_total_rewards()
                if len(set(rewards)) <= 1: continue
                trajectories=getattr(group,'trajectories_G',None)
                if trajectories is not None and not any(len(t.ac.tokens) and not t.metrics.get('parse_error_masked') for trajectory in trajectories for t in trajectory.transitions):
                    continue
                retained.append((builder,group))
            if retained:
                bound.arguments['env_group_builders_P']=[b for b,_ in retained]
                bound.arguments['trajectory_groups_P']=[g for _,g in retained]
                return await group_step(*bound.args,**bound.kwargs)
            # Checkpointing unchanged weights maintains the sampler/checkpoint contract;
            # no forward/backward or Adam call occurs, including momentum updates.
            client,metrics=await train_module.save_checkpoint_and_get_sampling_client(
                bound.arguments['training_client'],bound.arguments['checkpoint_mgr'],bound.arguments['i_batch']+1)
            metrics['workshop/zero_signal_batch']=1
            record(0,0,0,metrics)
            return client,metrics
        train_module.do_train_step_and_get_sampling_client = filtered_step
    return counter
