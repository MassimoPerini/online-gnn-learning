import numpy as np
import random
from prioritized_replay.segment_tree import SumSegmentTree
import math


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, node_id):
        data = node_id
        self._storage.append(data)
        self._next_idx = self._next_idx + 1


    def _encode_sample(self, idxes):
        obses_t = []
        for i in idxes:
            data = self._storage[i]
            obses_t.append(np.array(data, copy=False))

        return np.array(obses_t)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha, max_priority, min_priority):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - proportional prioritization)
        max_priority: float
            defines the maximum value of priority. Higher values will be clipped
        min_priority: float
            defines the minimum value of priority. Lower values will be clipped
        start_priority: float
            defines the priorities starting value
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._max_clip_priority = max_priority
        self._min_clip_priority = min_priority
        self._key_to_idx = {}
        self._init_min_max()

        print("PrioritizedBuffer init, alpha: ", self._alpha)

    def _init_min_max(self):
        self._max_priority = -1
        self._min_priority = 99999999

        self.max_val = -1
        self.min_val = 99999999

    def get_max_priority(self):
        return self.max_val

    def get_min_priority(self):
        return self.min_val


    def _normalize(self, node_priority_dict):
        d_priorities = {}

        for node, priority in node_priority_dict.items():

            priority = (min(max(priority, self._min_clip_priority), self._max_clip_priority))

            if priority > self.max_val:
                self.max_val = priority
            if priority < self.min_val:
                self.min_val = priority

            priority = math.log(priority)
            d_priorities[node] = priority

            if priority > self._max_priority:
                self._max_priority = priority
            if priority < self._min_priority:
                self._min_priority = priority

        return d_priorities


    def add_all(self, node_priority_dict):
        '''
        add vertices to the structure

        :param node_priority_dict: dict of vertex: priority
        :return:
        '''

        d_priorities = self._normalize(node_priority_dict)
        scale = (self._max_priority - self._min_priority)
        res = {}

        for node, priority in d_priorities.items():
            #print(priority, " ", self._min_priority, " ", scale)
            if scale > 0:
                v = (priority - self._min_priority) / scale
            else:
                v = (priority - self._min_priority)

            v += 0.00001

            idx = self._next_idx
            super().add(node)
            self._key_to_idx[node] = idx
            self._it_sum[idx] = v ** self._alpha
            #print(v)
            assert(v >= 0)
            res[node] = v



    def _sample_proportional(self, batch_size):

        if batch_size >= len(self._storage):
            return list(self._key_to_idx.values())

        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        sampled_weights = []
        #cnt_below = 0


        res = set()

        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.add(idx)
            sampled_weights += [self._it_sum[idx]]
            #if self._it_sum[idx] < 0.2:
            #    cnt_below += 1

        j = 0

        while len(res) < batch_size:
            mass = random.random() * p_total
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.add(idx)
            sampled_weights += [self._it_sum[idx]]
            #print(len(res))
            j += 1
            if j > 20:#maybe I'm stuck sampling the same item
                break


        missing_els = batch_size - len(res)
        while missing_els > 0:#draw randomly
            res.add(random.randint(0, len(self._storage) - 1))
            missing_els = batch_size - len(res)

        return res

    def sample(self, batch_size):
        """Sample a batch of vertices.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        list of vertices
        """
        idxes = self._sample_proportional(batch_size)
        encoded_sample = self._encode_sample(idxes)
        return list(encoded_sample)

    def update_priorities(self, d_priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        d_priorities: [int (vertex id), float (priority value)]
            dict of updated priorities corresponding to
            vertices
        """

        d_priorities = self._normalize(d_priorities)
        scale = (self._max_priority - self._min_priority)
        #prnt = {}

        for node, priority in d_priorities.items():

            if scale > 0:
                v = (priority - self._min_priority) / scale
            else:
                v = (priority - self._min_priority)

            v += 0.000001
            assert(v >= 0)
            idx = self._key_to_idx[node]
            self._it_sum[idx] = v ** self._alpha
            #prnt[node] = v

        #print("UPDATE: ", prnt)

    def increment_priorities(self, node, increment):
        """Increase the priority of the given node.
        Parameters
        ----------
        node: [int]
            Vertex
        priorities: [float]
            Increment of priority.
        """
        assert increment >= 0

        idx = self._key_to_idx[node]

        diff = self._max_priority - self._min_priority

        if self._max_priority == -1:
            self._it_sum[idx] += (increment ** self._alpha)
            self._it_sum[idx] = min(self._it_sum[idx], 1) #normalized between 0 and 1
        else:
            self._it_sum[idx] += (increment*diff)
            self._it_sum[idx] = min(self._it_sum[idx], 1)

        #if self._it_sum[idx] > self.max_val:
        #    self.max_val = self._it_sum[idx]


    def dump_priorities(self, vertex_list):
        result = []
        for vertex in vertex_list:
            r = self._it_sum[self._key_to_idx[vertex]]
            #assert(self._min_priority <= r <= self._max_priority)
            result += [r]

        return result

