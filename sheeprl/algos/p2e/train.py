from torch.distributions import Independent, Normal

from sheeprl.algos.p2e.loss import ensemble_loss


def ensemble_learning(fabric, ensembles, ensemble_optims, states, actions, embedded_obs, aggregator, args):
    # Ensemble

    for i, ens in enumerate(ensembles):
        ensemble_optims[i].zero_grad(set_to_none=True)
        out = ens(states.detach(), actions.detach())
        embeds_dist = Independent(Normal(out, 1), 1)
        loss = ensemble_loss(embeds_dist, embedded_obs.detach())
        loss.backward()
        fabric.clip_gradients(module=ens, optimizer=ensemble_optims[i], max_norm=args.ensemble_clip_gradients)
        ensemble_optims[i].step()
        aggregator.update(f"Ensemble/loss_{i}", loss.detach().cpu())
