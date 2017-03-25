__author__ = 'SongJun-Dell'

class CheckerGraident(object):
    def __init__(self):
        self.delta = 1e-5
        self.rel_error_thr_warning = 1e-2
        self.rel_error_thr_error = 1.0

        self.status_types = ['PASS', 'ZEOR', 'SMALL', 'WARNING', '!!!---ERROR---!!!']
        self.status_counts = [0, 0, 0, 0, 0]
        self.status_params = [[], [], [], [], []]

    def check_gradient(self):
        cost, grads_weights = self.get_loss(mode='gc')
        gc_params = self.get_gc_params()

        for p in gc_params.keys():
            p_status = 0
            status_counts = [ 0 for i in xrange(len(self.status_types))]
            param_mat = gc_params[p]
            print 'checking gradient on parameter %s of shape %s.' % (p, str(param_mat.shape))
            assert param_mat.shape == grads_weights[p].shape, 'Error, dims do not match: %s and %s.' % \
                                                            (str(param_mat.shape), str(grads_weights[p].shape))

            for i in xrange(param_mat.size):
                # evaluate cost at [x+delta]
                param_mat.flat[i] += self.delta
                new_cost = self.get_loss(mode='test')
                param_mat.flat[i] -= self.delta

                # compare numerical and analytic grads
                grad_numerical = (new_cost-cost)/self.delta
                grad_analytic = grads_weights[p].flat[i]
                v_status = 0 # pass
                if grad_analytic == 0 and grad_numerical == 0:
                    rel_error = 0  # both are zero, ok
                    v_status = 1 # zero
                    status_counts[1] += 1
                elif abs(grad_numerical) < 1e-7 and abs(grad_analytic) < 1e-7:
                    rel_error = 0  # not enough precision to check this
                    v_status = 2 # vel small
                    status_counts[2] += 1
                else:
                    rel_error = abs(grad_analytic-grad_numerical) / abs(grad_numerical+grad_analytic)
                    if rel_error > self.rel_error_thr_error:
                        v_status = 4
                        status_counts[4] += 1
                    elif rel_error > self.rel_error_thr_warning:
                        v_status = 3
                        status_counts[3] += 1
                    else:
                        v_status = 0
                        status_counts[0] += 1

                p_status = max(p_status, v_status)
                # print status
                detail = True
                if v_status > 0 and detail:
                    print '%s\t%s\tindex\t%4d (val = %+8f), analytic= %+8f, numerical= %+8f, relative error= %+8f' \
                    % (p, self.status_types[v_status], i, param_mat.flat[i], grad_analytic, grad_numerical, rel_error)
            status_count_meg = ''
            for ss, sc in zip(self.status_types, status_counts):
                if sc > 0:
                    status_count_meg += '%s: %d, ' % (ss, sc)
            print '[%s] checking gradient on parameter %s of shape %s, %s.' % (self.status_types[p_status], p, str(param_mat.shape), status_count_meg)
            self.status_counts[p_status] += 1
            self.status_params[p_status].append(p)

        for i in xrange(len(self.status_types)):
            print '%s parameters: %d' % (self.status_types[i], self.status_counts[i])
            print '\n'.join(self.status_params[i])
            print '\n'

    def get_loss(self, mode):
        """
        if mode=='train':
            return loss, grad_params
        elif mode=='test':
            return loss
        """
        raise NotImplementedError

    def get_gc_params(self):
        """
        :return:
        gc_params = dict()
        """
        raise NotImplementedError
