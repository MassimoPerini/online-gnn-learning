import numpy as np

class GeneratePriority():
    def get_priorities(self, batch_nodes_seed, losses):
        raise NotImplementedError

class LossPriority(GeneratePriority):
    def get_priorities(self, batch_nodes_seed, losses):
        return losses

class TrendPriority(GeneratePriority):
    def __init__(self, n_vertices, alpha = 0.85):
        self.values = np.zeros(n_vertices, dtype=np.float)
        self.prev_loss = np.zeros(n_vertices, dtype=np.float)
        self.init = np.full(n_vertices, True, dtype=bool)

        self.avg = 0
        self.n_items = 0

        self.alpha = alpha

    def get_priorities(self, batch_nodes_seed, losses):

        filter = self.init[batch_nodes_seed]
        new_vertices = np.asarray(batch_nodes_seed)[filter]
        self.init[new_vertices] = False

        self.values[new_vertices] = self.avg #todo
        self.n_items += len(new_vertices)


        update = losses - self.prev_loss[batch_nodes_seed]
        update = update.clip(min=0)

        #cumulate
        self.avg *= self.n_items
        self.avg -= np.sum(self.values[batch_nodes_seed])

        self.values[batch_nodes_seed] *= self.alpha
        self.values[batch_nodes_seed] += (update*(1-self.alpha))

        self.avg += np.sum(self.values[batch_nodes_seed])
        self.avg /= self.n_items

        self.prev_loss[batch_nodes_seed] = losses
        return self.values[batch_nodes_seed]


class HybridPriority(GeneratePriority):
    def __init__(self, n_vertices, alpha = 0.85, loss_contrib = 0.5):
        self.trend_p = TrendPriority(n_vertices, alpha)
        self.loss_p = LossPriority()
        self.loss_contrib = loss_contrib

    def get_priorities(self, batch_nodes_seed, losses):
        prior = self.trend_p.get_priorities(batch_nodes_seed, losses) * (1-self.loss_contrib)
        prior += (self.loss_p.get_priorities(batch_nodes_seed, losses) * self.loss_contrib)
        return prior
