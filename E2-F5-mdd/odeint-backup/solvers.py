import abc
import torch
from .event_handling import find_event
from .misc import _handle_unused_kwargs
import sys
import pdb
from tqdm import trange, tqdm


class AdaptiveStepsizeODESolver(metaclass=abc.ABCMeta):
    def __init__(self, dtype, y0, norm, **unused_kwargs):
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.y0 = y0
        self.dtype = dtype

        self.norm = norm

    def _before_integrate(self, t):
        pass

    @abc.abstractmethod
    def _advance(self, next_t):
        raise NotImplementedError

    @classmethod
    def valid_callbacks(cls):
        return set()

    def integrate(self, t):
        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0
        t = t.to(self.dtype)
        self._before_integrate(t)
        for i in range(1, len(t)):
            solution[i] = self._advance(t[i])
        return solution


class AdaptiveStepsizeEventODESolver(AdaptiveStepsizeODESolver, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def _advance_until_event(self, event_fn):
        raise NotImplementedError

    def integrate_until_event(self, t0, event_fn):
        t0 = t0.to(self.y0.device, self.dtype)
        self._before_integrate(t0.reshape(-1))
        event_time, y1 = self._advance_until_event(event_fn)
        solution = torch.stack([self.y0, y1], dim=0)
        return event_time, solution


class FixedGridODESolver(metaclass=abc.ABCMeta):
    order: int

    def __init__(self, func, y0, step_size=None, grid_constructor=None, interp="linear", perturb=False, **unused_kwargs):
        self.atol = unused_kwargs.pop('atol')
        unused_kwargs.pop('rtol', None)
        unused_kwargs.pop('norm', None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.dtype = y0.dtype
        self.device = y0.device
        self.step_size = step_size
        self.interp = interp
        self.perturb = perturb

        if step_size is None:
            if grid_constructor is None:
                self.grid_constructor = lambda f, y0, t: t
            else:
                self.grid_constructor = grid_constructor
        else:
            if grid_constructor is None:
                self.grid_constructor = self._grid_constructor_from_step_size(step_size)
            else:
                raise ValueError("step_size and grid_constructor are mutually exclusive arguments.")

    @classmethod
    def valid_callbacks(cls):
        return {'callback_step'}

    @staticmethod
    def _grid_constructor_from_step_size(step_size):
        def _grid_constructor(func, y0, t):
            start_time = t[0]
            end_time = t[-1]

            niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = torch.arange(0, niters, dtype=t.dtype, device=t.device) * step_size + start_time
            t_infer[-1] = t[-1]

            return t_infer
        return _grid_constructor

    @abc.abstractmethod
    def _step_func(self, func, t0, dt, t1, y0):
        pass

    def integrate(self, t):
        time_grid = self.grid_constructor(self.func, self.y0, t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0

        j = 1
        y0 = self.y0
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dt = t1 - t0
            self.func.callback_step(t0, y0, dt)
            dy, f0 = self._step_func(self.func, t0, dt, t1, y0)
            y1 = y0 + dy

            while j < len(t) and t1 >= t[j]:
                if self.interp == "linear":
                    solution[j] = self._linear_interp(t0, t1, y0, y1, t[j])
                elif self.interp == "cubic":
                    f1 = self.func(t1, y1)
                    solution[j] = self._cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t[j])
                else:
                    raise ValueError(f"Unknown interpolation method {self.interp}")
                j += 1
            y0 = y1

        return solution

    def integrate_until_event(self, t0, event_fn):
        assert self.step_size is not None, "Event handling for fixed step solvers currently requires `step_size` to be provided in options."

        t0 = t0.type_as(self.y0.abs())
        y0 = self.y0
        dt = self.step_size

        sign0 = torch.sign(event_fn(t0, y0))
        max_itrs = 20000
        itr = 0
        while True:
            itr += 1
            t1 = t0 + dt
            dy, f0 = self._step_func(self.func, t0, dt, t1, y0)
            y1 = y0 + dy

            sign1 = torch.sign(event_fn(t1, y1))

            if sign0 != sign1:
                if self.interp == "linear":
                    interp_fn = lambda t: self._linear_interp(t0, t1, y0, y1, t)
                elif self.interp == "cubic":
                    f1 = self.func(t1, y1)
                    interp_fn = lambda t: self._cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t)
                else:
                    raise ValueError(f"Unknown interpolation method {self.interp}")
                event_time, y1 = find_event(interp_fn, sign0, t0, t1, event_fn, float(self.atol))
                break
            else:
                t0, y0 = t1, y1

            if itr >= max_itrs:
                raise RuntimeError(f"Reached maximum number of iterations {max_itrs}.")
        solution = torch.stack([self.y0, y1], dim=0)
        return event_time, solution

    def _cubic_hermite_interp(self, t0, y0, f0, t1, y1, f1, t):
        h = (t - t0) / (t1 - t0)
        h00 = (1 + 2 * h) * (1 - h) * (1 - h)
        h10 = h * (1 - h) * (1 - h)
        h01 = h * h * (3 - 2 * h)
        h11 = h * h * (h - 1)
        dt = (t1 - t0)
        return h00 * y0 + h10 * dt * f0 + h01 * y1 + h11 * dt * f1

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)


##XINWEI: for MDD
class FixedGridODESolverJACOBTRACE(metaclass=abc.ABCMeta):
    order: int

    def __init__(self, func, y0, step_size=None, grid_constructor=None, interp="linear", perturb=False, **unused_kwargs):
        self.atol = unused_kwargs.pop('atol')
        unused_kwargs.pop('rtol', None)
        unused_kwargs.pop('norm', None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.dtype = y0.dtype
        self.device = y0.device
        self.step_size = step_size
        self.interp = interp
        self.perturb = perturb

        if step_size is None:
            if grid_constructor is None:
                self.grid_constructor = lambda f, y0, t: t
            else:
                self.grid_constructor = grid_constructor
        else:
            if grid_constructor is None:
                self.grid_constructor = self._grid_constructor_from_step_size(step_size)
            else:
                raise ValueError("step_size and grid_constructor are mutually exclusive arguments.")

    @classmethod
    def valid_callbacks(cls):
        return {'callback_step'}

    @staticmethod
    def _grid_constructor_from_step_size(step_size):
        def _grid_constructor(func, y0, t):
            start_time = t[0]
            end_time = t[-1]

            niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = torch.arange(0, niters, dtype=t.dtype, device=t.device) * step_size + start_time
            t_infer[-1] = t[-1]

            return t_infer
        return _grid_constructor

    @abc.abstractmethod
    def _step_func(self, func, t0, dt, t1, y0):
        pass

    def integrate(self, t, cond_mask, t_interval, cfg): ##cond_mask is B x T
        time_grid = self.grid_constructor(self.func, self.y0, t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]
        ##XINWEI: MDD EXTRA assertion, no interpolation
        assert len(t) == len(time_grid)       
        b_size = self.y0.shape[0]
        t_size = self.y0.shape[1]
        d_size = self.y0.shape[-1]
        if cfg == 0:
            if t_size < 300:
                grad_batch = 100
            elif t_size < 500:
                grad_batch = 50
            elif t_size < 700:
                grad_batch = 25
            else:
                grad_batch = 10  
        else:
            if t_size < 250:
                grad_batch = 100
            elif t_size < 350:
                grad_batch = 50
            elif t_size < 450:
                grad_batch = 25
            else:
                grad_batch = 10
        
        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        jacob_trace = torch.empty(len(t), *self.y0.shape[:-1], dtype=self.y0.dtype, device=self.y0.device)
        g_out = torch.eye(d_size, device=self.y0.device).repeat((b_size, t_size, 1, 1)) # B x T X D x D Tensor 
        g_out[cond_mask] = 0
        solution[0] = self.y0
        with torch.enable_grad():
            #self.y0.requires_grad_(True)
            y0 = self.y0.clone()
            iterator = tqdm(zip(time_grid[:-1], time_grid[1:]), desc="ODE MDD")
            #b_trace=torch.vmap(torch.trace) 
            for i, (t0, t1) in enumerate(iterator):
                y0.requires_grad_(True)
                dt = t1 - t0         
                dy, f0 = self._step_func(self.func, t0, dt, t1, y0)
                def get_trace(v_t):
                    temp_grad = torch.autograd.grad(f0, y0, v_t, retain_graph=True) ##  B x T x D Tensor
                    return temp_grad
                get_trace_batch = torch.vmap(get_trace, in_dims=2, out_dims=2, chunk_size=grad_batch)
                for t in range(t_interval):
                    t_mask_single = torch.arange(start=t, end=t_size, step=t_interval, device=self.y0.device)         
                    t_mask = torch.zeros_like(g_out, dtype=torch.bool, device=self.y0.device)
                    t_mask[:,t_mask_single,:,:] = True
                    results = get_trace_batch(g_out*t_mask) ##  B x T x D x D Tensor
                    #negate for forward
                    jacob_trace[i,:,t_mask_single] = torch.diagonal(-results[0][:,t_mask_single],  dim1=-2, dim2=-1).sum(-1)
                y1 = y0.detach() + dy.detach()
                solution[i+1] = y1
                del dy,f0
                torch.cuda.empty_cache() 
                y0 = y1      
                               
            ## final step jacobian needed for MDD
            y0.requires_grad_(True)         
            dy, f0 = self._step_func(self.func, t1, 1, None, y0)          
            def get_trace(v_t):
                temp_grad = torch.autograd.grad(f0, y0, v_t, retain_graph=True) ##  B x DT Tensor
                return temp_grad#return BxT Tensor
            get_trace_batch = torch.vmap(get_trace, in_dims=2, out_dims=2, chunk_size=grad_batch)
            for t in range(t_interval):             
                t_mask_single = torch.arange(start=t, end=t_size, step=t_interval, device=self.y0.device)             
                t_mask = torch.zeros_like(g_out, dtype=torch.bool, device=self.y0.device)
                t_mask[:,t_mask_single,:,:] = True
                results = get_trace_batch(g_out*t_mask) ##  B x T x D x D Tensor
                jacob_trace[i+1,:,t_mask_single] = torch.diagonal(-results[0][:,t_mask_single],  dim1=-2, dim2=-1).sum(-1)
            del dy,f0
            torch.cuda.empty_cache() 
            return solution, jacob_trace
    ###not good
    # def integrate(self, t):
    #     time_grid = self.grid_constructor(self.func, self.y0, t)
    #     assert time_grid[0] == t[0] and time_grid[-1] == t[-1]
    #     ##XINWEI: MDD EXTRA assertion, no interpolation
    #     assert len(t) == len(time_grid)

    #     solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
    #     jacob_trace = torch.empty(len(t), *self.y0.shape[:-1], dtype=self.y0.dtype, device=self.y0.device)
    #     solution[0] = self.y0

    #     b_size = self.y0.shape[0]
    #     t_size = self.y0.shape[1]
    #     d_size = self.y0.shape[-1]
    #     with torch.enable_grad():
    #         #self.y0.requires_grad_(True)
    #         y0 = self.y0.clone()
    #         iterator = tqdm(zip(time_grid[:-1], time_grid[1:]), desc="ODE MDD")
    #         #b_trace=torch.vmap(torch.trace)
    #         #for i, (t0, t1) in enumerate(zip(time_grid[:-1], time_grid[1:])):
    #         for i, (t0, t1) in enumerate(iterator):
    #             y0 = y0.reshape((b_size, t_size*d_size))
    #             y0.requires_grad_(True)
    #             y0_rs = y0.view((b_size, t_size, d_size))
    #             dt = t1 - t0             
    #             dy, f0 = self._step_func(self.func, t0, dt, t1, y0_rs)
    #             ## negate for MDD(forward), check the function _check_inputs in odeint.py
    #             f0 = -f0
    #             f0 = f0.reshape((b_size, t_size*d_size))                
    #             g_out = torch.eye(d_size*t_size, device=self.y0.device).expand((b_size, t_size*d_size, t_size*d_size)) # B x DT x DT Tensor 
    #             def get_trace(v_t):
    #                 temp_grad = torch.autograd.grad(f0, y0, v_t, retain_graph=True) ##  B x DT Tensor
    #                 return temp_grad#return BxT Tensor
    #             get_trace_batch = torch.vmap(get_trace, in_dims=1, out_dims=1, chunk_size=200)
    #             results = get_trace_batch(g_out) ##  B x DT X DT Tensor
    #             jacob_trace[i] = torch.diagonal(results[0],  dim1=-2, dim2=-1).view(b_size,t_size,d_size).sum(-1)
    #             torch.cuda.empty_cache()
    #             y1 = y0.detach().view((b_size,t_size,d_size)) + dy.detach().view(b_size,t_size,d_size)
    #             solution[i+1] = y1
    #             y0 = y1
    #             del dy,f0,y0_rs
    #             torch.cuda.empty_cache()         
    #         ## final step jacobian needed for MDD
    #         y0 = y0.reshape((b_size, t_size*d_size))
    #         y0.requires_grad_(True)
    #         y0_rs = y0.view((b_size, t_size, d_size))
    #         dt = t1 - t0             
    #         dy, f0 = self._step_func(self.func, t0, dt, t1, y0_rs)
    #         ## negate for MDD(forward), check the function _check_inputs in odeint.py
    #         f0 = -f0
    #         f0 = f0.reshape((b_size, t_size*d_size))                
    #         g_out = torch.eye(d_size*t_size, device=self.y0.device).expand((b_size, t_size*d_size, t_size*d_size)) # B x DT x DT Tensor 
    #         def get_trace(v_t):
    #             temp_grad = torch.autograd.grad(f0, y0, v_t, retain_graph=True) ##  B x DT Tensor
    #             return temp_grad#return BxT Tensor
    #         get_trace_batch = torch.vmap(get_trace, in_dims=1, out_dims=1, chunk_size=150)
    #         results = get_trace_batch(g_out) ##  B x DT X DT Tensor
    #         jacob_trace[i] = torch.diagonal(results[0],  dim1=-2, dim2=-1).view(b_size,t_size,d_size).sum(-1)
            
    #         return solution, jacob_trace

        ## wrong again
        # def integrate(self, t):
        #     time_grid = self.grid_constructor(self.func, self.y0, t)
        #     assert time_grid[0] == t[0] and time_grid[-1] == t[-1]
        #     ##XINWEI: MDD EXTRA assertion, no interpolation
        #     assert len(t) == len(time_grid)

        #     solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        #     jacob_trace = torch.empty(len(t), *self.y0.shape[:-1], dtype=self.y0.dtype, device=self.y0.device)
        #     solution[0] = self.y0

        #     b_size = self.y0.shape[0]
        #     t_size = self.y0.shape[1]
        #     d_size = self.y0.shape[-1]
        #     with torch.enable_grad():
        #         #self.y0.requires_grad_(True)
        #         y0 = self.y0.clone()
        #         iterator = tqdm(zip(time_grid[:-1], time_grid[1:]), desc="ODE MDD")
        #         #b_trace=torch.vmap(torch.trace)
        #         #for i, (t0, t1) in enumerate(zip(time_grid[:-1], time_grid[1:])):
        #         for i, (t0, t1) in enumerate(iterator):
        #             y0 = y0.reshape((b_size*t_size,d_size))
        #             y0.requires_grad_(True)
        #             y0_rs = y0.view((b_size, t_size, d_size))
        #             dt = t1 - t0             
        #             dy, f0 = self._step_func(self.func, t0, dt, t1, y0_rs)
        #             ## negate for MDD(forward), check the function _check_inputs in odeint.py
        #             f0 = -f0
        #             f0 = f0.reshape((b_size*t_size, d_size))                
        #             g_out = torch.eye(d_size, device=self.y0.device).expand((b_size*t_size, d_size, d_size)) # BT x D x D Tensor 
        #             def get_trace(v_t):
        #                 temp_grad = torch.autograd.grad(f0, y0, v_t, retain_graph=True) 
        #                 return temp_grad#return BTxD Tensor
        #             get_trace_batch = torch.vmap(get_trace, in_dims=1, out_dims=1)
        #             results = get_trace_batch(g_out) ##  BT x D X D Tensor
        #             pdb.set_trace()
        #             jacob_trace[i] = torch.diagonal(results[0],  dim1=-2, dim2=-1).view(b_size,t_size,d_size).sum(-1)
        #             torch.cuda.empty_cache()
        #             y1 = y0.detach().view((b_size,t_size,d_size)) + dy.detach().view(b_size,t_size,d_size)
        #             solution[i+1] = y1
        #             y0 = y1
        #             del dy,f0,y0_rs
        #             torch.cuda.empty_cache()         
        #         ## final step jacobian needed for MDD
        #         y0 = y0.reshape((b_size, t_size*d_size))
        #         y0.requires_grad_(True)
        #         y0_rs = y0.view((b_size, t_size, d_size))
        #         dt = t1 - t0             
        #         dy, f0 = self._step_func(self.func, t0, dt, t1, y0_rs)
        #         ## negate for MDD(forward), check the function _check_inputs in odeint.py
        #         f0 = -f0
        #         f0 = f0.reshape((b_size, t_size*d_size))                
        #         g_out = torch.eye(d_size*t_size, device=self.y0.device).expand((b_size, t_size*d_size, t_size*d_size)) # B x DT x DT Tensor 
        #         def get_trace(v_t):
        #             temp_grad = torch.autograd.grad(f0, y0, v_t, retain_graph=True) ##  B x DT Tensor
        #             return temp_grad#return BxT Tensor
        #         get_trace_batch = torch.vmap(get_trace, in_dims=1, out_dims=1, chunk_size=150)
        #         results = get_trace_batch(g_out) ##  B x DT X DT Tensor
        #         jacob_trace[i] = torch.diagonal(results[0],  dim1=-2, dim2=-1).view(b_size,t_size,d_size).sum(-1)
                
        #         return solution, jacob_trace

        #still too slow
        # y0_in = [y0[:,t,:].requires_grad_(True) for t in range(t_size)]
        # y0_in_stacked = torch.stack(y0_in,dim=1) ##stack will be in the graph
        # dy, f0 = self._step_func(self.func, t0, dt, t1, y0_in_stacked)
        # for t in range(t_size):
        #     f0_slice = f0[:,t,:]
        #     y0_slice = y0_in[t]
        #     def get_trace(v_t):
        #         temp_grad = torch.autograd.grad(f0_slice, y0_slice, v_t, retain_graph=True) ##  Bx1 Tensors
        #         return temp_grad#return BxT Tensor
        #     get_trace_batch = torch.vmap(get_trace, in_dims=1, out_dims=1)
        #     grad_out = torch.eye(d_size, device=self.y0.device)
        #     grad_out = grad_out.expand(b_size,d_size,d_size)
        #     grad_batched = get_trace_batch(grad_out) ## BxDxD Tensor
        #     sum[:,t] = b_trace(grad_batched[0])
        #f0_out = [ f0[:,t,:] for t in range(t_size)] ## slice will be now in the graph
        #f0_out_stacked = torch.stack(f0_out, dim=1) ## stacked tensors will be now in the graph, so batch can be applied. great trick!!!
        ## VMAP can be done for non-gradient args only!!!
        # sum = torch.zeros(b_size, t_size, device=self.y0.device)
        # for d in range(d_size):
        #     def get_trace(f_t, v_t):
        #         pdb.set_trace()
        #         temp_grad = torch.autograd.grad(f_t, y0_in, v_t, retain_graph=True) ## tuple of BxD Tensors
        #         return torch.stack(temp_grad, dim=1)[:,:,d] #return BxT Tensor
        #     #get_trace_batch = torch.vmap(get_trace, in_dims=(1,1), out_dims=(1,1)) #BxTx1
        #     get_trace_batch = torch.vmap(get_trace, in_dims=1, out_dims=1) #BxTx1
        #     grad_out = torch.zeros((b_size, t_size, d_size), device=self.y0.device)
        #     grad_out[:,:,d] = 1 
        #     pdb.set_trace()
        #     batched_out = get_trace_batch(f0_out_stacked, grad_out)  #BxTx1
        #     sum[:,:] += batched_out
        #     # for t in range(t_size):
        #     #    grad_seq = torch.autograd.grad(f0_out[t], y0_in[t], grad_out, retain_graph=True)[0] ## a tensor of B x D 
        #     #    sum[:,t] += grad_seq[:,d]
        #     print(f"{d}")
        # must do non-t-batch here, for the correct jacobian, too slow!!!
        # for t_index in range(t_size):
        #     f0_slice = f0[:,t_index,:]
        #     for j in range(self.y0.shape[-1]):
        #         grad_out = torch.zeros_like(f0_slice)
        #         grad_out[:,j] = 1 
        #         sum[:,t_index] += torch.autograd.grad(f0_slice, y0[t_index], grad_out, retain_graph=True)[0][:,j] ## B x 1
        #     print(f'{t_index}')

    def integrate_until_event(self, t0, event_fn):
        assert self.step_size is not None, "Event handling for fixed step solvers currently requires `step_size` to be provided in options."

        t0 = t0.type_as(self.y0.abs())
        y0 = self.y0
        dt = self.step_size

        sign0 = torch.sign(event_fn(t0, y0))
        max_itrs = 20000
        itr = 0
        while True:
            itr += 1
            t1 = t0 + dt
            dy, f0 = self._step_func(self.func, t0, dt, t1, y0)
            y1 = y0 + dy

            sign1 = torch.sign(event_fn(t1, y1))

            if sign0 != sign1:
                if self.interp == "linear":
                    interp_fn = lambda t: self._linear_interp(t0, t1, y0, y1, t)
                elif self.interp == "cubic":
                    f1 = self.func(t1, y1)
                    interp_fn = lambda t: self._cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t)
                else:
                    raise ValueError(f"Unknown interpolation method {self.interp}")
                event_time, y1 = find_event(interp_fn, sign0, t0, t1, event_fn, float(self.atol))
                break
            else:
                t0, y0 = t1, y1

            if itr >= max_itrs:
                raise RuntimeError(f"Reached maximum number of iterations {max_itrs}.")
        solution = torch.stack([self.y0, y1], dim=0)
        return event_time, solution

    def _cubic_hermite_interp(self, t0, y0, f0, t1, y1, f1, t):
        h = (t - t0) / (t1 - t0)
        h00 = (1 + 2 * h) * (1 - h) * (1 - h)
        h10 = h * (1 - h) * (1 - h)
        h01 = h * h * (3 - 2 * h)
        h11 = h * h * (h - 1)
        dt = (t1 - t0)
        return h00 * y0 + h10 * dt * f0 + h01 * y1 + h11 * dt * f1

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)
    
    
##XINWEI: for MDD
class FixedGridODESolverJACOBTRACE_Wrong(metaclass=abc.ABCMeta):
    order: int

    def __init__(self, func, y0, step_size=None, grid_constructor=None, interp="linear", perturb=False, **unused_kwargs):
        self.atol = unused_kwargs.pop('atol')
        unused_kwargs.pop('rtol', None)
        unused_kwargs.pop('norm', None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.dtype = y0.dtype
        self.device = y0.device
        self.step_size = step_size
        self.interp = interp
        self.perturb = perturb

        if step_size is None:
            if grid_constructor is None:
                self.grid_constructor = lambda f, y0, t: t
            else:
                self.grid_constructor = grid_constructor
        else:
            if grid_constructor is None:
                self.grid_constructor = self._grid_constructor_from_step_size(step_size)
            else:
                raise ValueError("step_size and grid_constructor are mutually exclusive arguments.")

    @classmethod
    def valid_callbacks(cls):
        return {'callback_step'}

    @staticmethod
    def _grid_constructor_from_step_size(step_size):
        def _grid_constructor(func, y0, t):
            start_time = t[0]
            end_time = t[-1]

            niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = torch.arange(0, niters, dtype=t.dtype, device=t.device) * step_size + start_time
            t_infer[-1] = t[-1]

            return t_infer
        return _grid_constructor

    @abc.abstractmethod
    def _step_func(self, func, t0, dt, t1, y0):
        pass

    def integrate(self, t, cond_mask, cfg):
        time_grid = self.grid_constructor(self.func, self.y0, t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]
        ##XINWEI: MDD EXTRA assertion, no interpolation
        assert len(t) == len(time_grid)
        
        b_size = self.y0.shape[0]
        t_size = self.y0.shape[1]
        d_size = self.y0.shape[-1]
        if cfg == 0:
            if t_size < 300:
                grad_batch = 100
            elif t_size < 500:
                grad_batch = 50
            elif t_size < 700:
                grad_batch = 25
            else:
                grad_batch = 10          
        else:
            if t_size < 250:
                grad_batch = 100
            elif t_size < 350:
                grad_batch = 50
            elif t_size < 420:
                grad_batch = 25
            elif t_size < 520:
                grad_batch = 10
            else:
                grad_batch = 4
            
        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        jacob_trace = torch.empty(len(t), *self.y0.shape[:-1], dtype=self.y0.dtype, device=self.y0.device)
        g_out = torch.eye(d_size, device=self.y0.device).repeat((b_size, t_size, 1, 1)) # B x T X D x D Tensor 
        g_out[cond_mask] = 0
        solution[0] = self.y0

        with torch.enable_grad():
            #self.y0.requires_grad_(True)
            y0 = self.y0.clone()
            #iterator = tqdm(zip(time_grid[:-1], time_grid[1:]), desc="ODE MDD")
            iterator = zip(time_grid[:-1], time_grid[1:])
            for i, (t0, t1) in enumerate(iterator):
                y0.requires_grad_(True)
                dt = t1 - t0  ## always greater than 0
                dy, f0 = self._step_func(self.func, t0, dt, None, y0)
                def get_trace(v_t):
                    temp_grad = torch.autograd.grad(f0, y0, v_t, retain_graph=True) ##  B x T x D Tensor
                    return temp_grad
                get_trace_batch = torch.vmap(get_trace, in_dims=2, out_dims=2, chunk_size=grad_batch)
                results = get_trace_batch(g_out) ##  B x T x D x D Tensor
                #negate for forward
                jacob_trace[i] = torch.diagonal(-results[0],  dim1=-2, dim2=-1).sum(-1) 
                y1 = y0.detach() + dy.detach()
                solution[i+1] = y1
                del dy,f0
                torch.cuda.empty_cache()
                y0 = y1
            ## final step jacobian needed for MDD
            y0.requires_grad_(True)
            dy, f0 = self._step_func(self.func, t1, 1, None, y0) 
            def get_trace(v_t):
                temp_grad = torch.autograd.grad(f0, y0, v_t, retain_graph=True) ##  B x T x D Tensor
                return temp_grad
            get_trace_batch = torch.vmap(get_trace, in_dims=2, out_dims=2, chunk_size=grad_batch)
            results = get_trace_batch(g_out) ##  B x T x D x D Tensor
            #negate for forward  ------ ??? why????
            jacob_trace[i+1] = torch.diagonal(-results[0],  dim1=-2, dim2=-1).sum(-1)
            del dy,f0
            torch.cuda.empty_cache()

        return solution, jacob_trace

    def integrate_until_event(self, t0, event_fn):
        assert self.step_size is not None, "Event handling for fixed step solvers currently requires `step_size` to be provided in options."

        t0 = t0.type_as(self.y0.abs())
        y0 = self.y0
        dt = self.step_size

        sign0 = torch.sign(event_fn(t0, y0))
        max_itrs = 20000
        itr = 0
        while True:
            itr += 1
            t1 = t0 + dt
            dy, f0 = self._step_func(self.func, t0, dt, t1, y0)
            y1 = y0 + dy

            sign1 = torch.sign(event_fn(t1, y1))

            if sign0 != sign1:
                if self.interp == "linear":
                    interp_fn = lambda t: self._linear_interp(t0, t1, y0, y1, t)
                elif self.interp == "cubic":
                    f1 = self.func(t1, y1)
                    interp_fn = lambda t: self._cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t)
                else:
                    raise ValueError(f"Unknown interpolation method {self.interp}")
                event_time, y1 = find_event(interp_fn, sign0, t0, t1, event_fn, float(self.atol))
                break
            else:
                t0, y0 = t1, y1

            if itr >= max_itrs:
                raise RuntimeError(f"Reached maximum number of iterations {max_itrs}.")
        solution = torch.stack([self.y0, y1], dim=0)
        return event_time, solution

    def _cubic_hermite_interp(self, t0, y0, f0, t1, y1, f1, t):
        h = (t - t0) / (t1 - t0)
        h00 = (1 + 2 * h) * (1 - h) * (1 - h)
        h10 = h * (1 - h) * (1 - h)
        h01 = h * h * (3 - 2 * h)
        h11 = h * h * (h - 1)
        dt = (t1 - t0)
        return h00 * y0 + h10 * dt * f0 + h01 * y1 + h11 * dt * f1

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)
    
##XINWEI: for MDD
class FixedGridODESolverDist(metaclass=abc.ABCMeta):
    order: int
    def __init__(self, func, y0, step_size=None, grid_constructor=None, interp="linear", perturb=False, **unused_kwargs):
        self.atol = unused_kwargs.pop('atol')
        unused_kwargs.pop('rtol', None)
        unused_kwargs.pop('norm', None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.dtype = y0.dtype
        self.device = y0.device
        self.step_size = step_size
        self.interp = interp
        self.perturb = perturb

        if step_size is None:
            if grid_constructor is None:
                self.grid_constructor = lambda f, y0, t: t
            else:
                self.grid_constructor = grid_constructor
        else:
            if grid_constructor is None:
                self.grid_constructor = self._grid_constructor_from_step_size(step_size)
            else:
                raise ValueError("step_size and grid_constructor are mutually exclusive arguments.")

    @classmethod
    def valid_callbacks(cls):
        return {'callback_step'}

    @staticmethod
    def _grid_constructor_from_step_size(step_size):
        def _grid_constructor(func, y0, t):
            start_time = t[0]
            end_time = t[-1]

            niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = torch.arange(0, niters, dtype=t.dtype, device=t.device) * step_size + start_time
            t_infer[-1] = t[-1]

            return t_infer
        return _grid_constructor

    @abc.abstractmethod
    def _step_func(self, func, t0, dt, t1, y0):
        pass

    def integrate(self, t):
        time_grid = self.grid_constructor(self.func, self.y0, t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]
        ##XINWEI: MDD EXTRA assertion, no interpolation
        assert len(t) == len(time_grid)
        
        b_size = self.y0.shape[0]
        t_size = self.y0.shape[1]
        d_size = self.y0.shape[-1]

        step_dist = torch.zeros(b_size, t_size, dtype=self.y0.dtype, device=self.y0.device)

        y0 = self.y0
        iterator = tqdm(zip(time_grid[:-1], time_grid[1:]), desc="ODE MDD")
        for i, (t0, t1) in enumerate(iterator):
            dt = t1 - t0  ## always greater than 0
            dy, f0 = self._step_func(self.func, t0, dt, None, y0)
            y1 = y0 + dy
            ##compute norm
            step_dist += torch.norm(dy, p=2, dim=-1)
            y0 = y1             
        ##compute optimal-transport distance
        #opt_dist=torch.cdist(y1[...,None,:], self.y0[...,None,:]).squeeze()
        opt_dist=torch.norm(y1 - self.y0, p=2, dim=-1)
        return opt_dist, step_dist

    def integrate_until_event(self, t0, event_fn):
        assert self.step_size is not None, "Event handling for fixed step solvers currently requires `step_size` to be provided in options."

        t0 = t0.type_as(self.y0.abs())
        y0 = self.y0
        dt = self.step_size

        sign0 = torch.sign(event_fn(t0, y0))
        max_itrs = 20000
        itr = 0
        while True:
            itr += 1
            t1 = t0 + dt
            dy, f0 = self._step_func(self.func, t0, dt, t1, y0)
            y1 = y0 + dy

            sign1 = torch.sign(event_fn(t1, y1))

            if sign0 != sign1:
                if self.interp == "linear":
                    interp_fn = lambda t: self._linear_interp(t0, t1, y0, y1, t)
                elif self.interp == "cubic":
                    f1 = self.func(t1, y1)
                    interp_fn = lambda t: self._cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t)
                else:
                    raise ValueError(f"Unknown interpolation method {self.interp}")
                event_time, y1 = find_event(interp_fn, sign0, t0, t1, event_fn, float(self.atol))
                break
            else:
                t0, y0 = t1, y1

            if itr >= max_itrs:
                raise RuntimeError(f"Reached maximum number of iterations {max_itrs}.")
        solution = torch.stack([self.y0, y1], dim=0)
        return event_time, solution

    def _cubic_hermite_interp(self, t0, y0, f0, t1, y1, f1, t):
        h = (t - t0) / (t1 - t0)
        h00 = (1 + 2 * h) * (1 - h) * (1 - h)
        h10 = h * (1 - h) * (1 - h)
        h01 = h * h * (3 - 2 * h)
        h11 = h * h * (h - 1)
        dt = (t1 - t0)
        return h00 * y0 + h10 * dt * f0 + h01 * y1 + h11 * dt * f1

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)
    
##XINWEI: for MDD
class FixedGridODESolverAABB(metaclass=abc.ABCMeta):
    order: int
    def __init__(self, func, y0, step_size=None, grid_constructor=None, interp="linear", perturb=False, **unused_kwargs):
        self.atol = unused_kwargs.pop('atol')
        unused_kwargs.pop('rtol', None)
        unused_kwargs.pop('norm', None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.dtype = y0.dtype
        self.device = y0.device
        self.step_size = step_size
        self.interp = interp
        self.perturb = perturb

        if step_size is None:
            if grid_constructor is None:
                self.grid_constructor = lambda f, y0, t: t
            else:
                self.grid_constructor = grid_constructor
        else:
            if grid_constructor is None:
                self.grid_constructor = self._grid_constructor_from_step_size(step_size)
            else:
                raise ValueError("step_size and grid_constructor are mutually exclusive arguments.")

    @classmethod
    def valid_callbacks(cls):
        return {'callback_step'}

    @staticmethod
    def _grid_constructor_from_step_size(step_size):
        def _grid_constructor(func, y0, t):
            start_time = t[0]
            end_time = t[-1]

            niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = torch.arange(0, niters, dtype=t.dtype, device=t.device) * step_size + start_time
            t_infer[-1] = t[-1]

            return t_infer
        return _grid_constructor

    @abc.abstractmethod
    def _step_func(self, func, t0, dt, t1, y0):
        pass

    def integrate(self, t):
        time_grid = self.grid_constructor(self.func, self.y0, t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]
        ##XINWEI: MDD EXTRA assertion, no interpolation
        assert len(t) == len(time_grid)
        
        b_size = self.y0.shape[0]
        t_size = self.y0.shape[1]
        d_size = self.y0.shape[-1]

        solution = torch.zeros(len(t)-1, *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        jacob_trace = torch.zeros(len(t)-1, *self.y0.shape[:-1], dtype=self.y0.dtype, device=self.y0.device)

        def aabb_estimate(input, t): ## in: B x T x D 
            eps = 3*torch.finfo(self.y0.dtype).eps  ## 3 because == operatior 
            in_plus = input[...,None,:].repeat(1,1,d_size,1) + torch.eye(d_size, dtype=self.y0.dtype, device=self.y0.device)*eps #B x T x D x D          
            in_minus = input[...,None,:].repeat(1,1,d_size,1) - torch.eye(d_size, dtype=self.y0.dtype, device=self.y0.device)*eps #B x T x D x D
            in_plus = input[None].repeat(d_size,1,1,1)  #D x B x T x D x D          
            in_minus = input[None].repeat(d_size,1,1,1) #D x B x T x D x D
            for i in range(d_size):
                in_plus[i,:,:,i]+=eps
                in_minus[i,:,:,i]-=eps
            jacob = ((self.func(t, in_plus.view(-1,t_size,d_size)) - self.func(t, in_minus.view(-1,t_size,d_size))) / 2*eps).reshape(d_size,*y0.shape)
            pdb.set_trace()
            trace = torch.diagonal(jacob.transpose(0,1).transpose(1,2), dim1=-2, dim2=-1).sum(-1)
            ##slow
            # trace = 0
            # for i in range(d_size): 
            #     in_plus = input.clone() 
            #     in_plus[:,:,i] += eps
            #     in_minus = input.clone()
            #     in_minus[:,:,i] -= eps
            #     pdb.set_trace()
            #     trace += (self.func(t, in_plus)[:,:,i] - self.func(t, in_minus)[:,:,i]) / 2*eps
            return trace            
        y0 = self.y0
        #iterator = tqdm(zip(time_grid[:-1], time_grid[1:]), desc="ODE MDD")
        iterator = zip(time_grid[:-1], time_grid[1:])
        ## from this version on, we only store the end-point jacobian and solution
        for i, (t0, t1) in enumerate(iterator):
            dt = t1 - t0  ## always greater than 0
            dy, f0 = self._step_func(self.func, t0, dt, None, y0)
            y1 = y0 + dy
            jacob_trace[i] = aabb_estimate(y1, t1)
            solution[i] = y1
            y0 = y1          
            
        return solution, jacob_trace

    def integrate_until_event(self, t0, event_fn):
        assert self.step_size is not None, "Event handling for fixed step solvers currently requires `step_size` to be provided in options."

        t0 = t0.type_as(self.y0.abs())
        y0 = self.y0
        dt = self.step_size

        sign0 = torch.sign(event_fn(t0, y0))
        max_itrs = 20000
        itr = 0
        while True:
            itr += 1
            t1 = t0 + dt
            dy, f0 = self._step_func(self.func, t0, dt, t1, y0)
            y1 = y0 + dy

            sign1 = torch.sign(event_fn(t1, y1))

            if sign0 != sign1:
                if self.interp == "linear":
                    interp_fn = lambda t: self._linear_interp(t0, t1, y0, y1, t)
                elif self.interp == "cubic":
                    f1 = self.func(t1, y1)
                    interp_fn = lambda t: self._cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t)
                else:
                    raise ValueError(f"Unknown interpolation method {self.interp}")
                event_time, y1 = find_event(interp_fn, sign0, t0, t1, event_fn, float(self.atol))
                break
            else:
                t0, y0 = t1, y1

            if itr >= max_itrs:
                raise RuntimeError(f"Reached maximum number of iterations {max_itrs}.")
        solution = torch.stack([self.y0, y1], dim=0)
        return event_time, solution

    def _cubic_hermite_interp(self, t0, y0, f0, t1, y1, f1, t):
        h = (t - t0) / (t1 - t0)
        h00 = (1 + 2 * h) * (1 - h) * (1 - h)
        h10 = h * (1 - h) * (1 - h)
        h01 = h * h * (3 - 2 * h)
        h11 = h * h * (h - 1)
        dt = (t1 - t0)
        return h00 * y0 + h10 * dt * f0 + h01 * y1 + h11 * dt * f1

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)
       
##XINWEI: for MDD
class FixedGridODESolverAABBFIX(metaclass=abc.ABCMeta):
    order: int
    def __init__(self, func, y0, step_size=None, grid_constructor=None, interp="linear", perturb=False, **unused_kwargs):
        self.atol = unused_kwargs.pop('atol')
        unused_kwargs.pop('rtol', None)
        unused_kwargs.pop('norm', None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.dtype = y0.dtype
        self.device = y0.device
        self.step_size = step_size
        self.interp = interp
        self.perturb = perturb

        if step_size is None:
            if grid_constructor is None:
                self.grid_constructor = lambda f, y0, t: t
            else:
                self.grid_constructor = grid_constructor
        else:
            if grid_constructor is None:
                self.grid_constructor = self._grid_constructor_from_step_size(step_size)
            else:
                raise ValueError("step_size and grid_constructor are mutually exclusive arguments.")

    @classmethod
    def valid_callbacks(cls):
        return {'callback_step'}

    @staticmethod
    def _grid_constructor_from_step_size(step_size):
        def _grid_constructor(func, y0, t):
            start_time = t[0]
            end_time = t[-1]

            niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = torch.arange(0, niters, dtype=t.dtype, device=t.device) * step_size + start_time
            t_infer[-1] = t[-1]

            return t_infer
        return _grid_constructor

    @abc.abstractmethod
    def _step_func(self, func, t0, dt, t1, y0):
        pass

    def integrate(self, t, t_interval):
        time_grid = self.grid_constructor(self.func, self.y0, t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]
        ##XINWEI: MDD EXTRA assertion, no interpolation
        assert len(t) == len(time_grid)
        
        b_size = self.y0.shape[0]
        t_size = self.y0.shape[1]
        d_size = self.y0.shape[-1]

        solution = torch.zeros(len(t)-1, *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        jacob_trace = torch.zeros(len(t)-1, *self.y0.shape[:-1], dtype=self.y0.dtype, device=self.y0.device)

        def aabb_estimate(input, t_in): ## in: B x T x D 
            eps = 3*torch.finfo(self.y0.dtype).eps  ## 3 because == operatior  
            trace = torch.zeros(*self.y0.shape[:-1], dtype=self.y0.dtype, device=self.y0.device)         
            for t in range(t_interval):
                t_index_single = torch.arange(start=t, end=t_size, step=t_interval, device=self.y0.device)  
                in_plus = input[None].repeat(d_size,1,1,1)  #D x B x T x D         
                in_minus = input[None].repeat(d_size,1,1,1) #D x B x T x D 
                for i in range(d_size):
                    in_plus[i,:,t_index_single,i]+=eps
                    in_minus[i,:,t_index_single,i]-=eps
                jacob = ((self.func(t_in, in_plus.view(-1,t_size,d_size)) - self.func(t_in, in_minus.view(-1,t_size,d_size))) / 2*eps).reshape(-1,*y0.shape)
                trace[:, t_index_single] = (torch.diagonal(jacob.transpose(0,1).transpose(1,2), dim1=-2, dim2=-1).sum(-1))[:,t_index_single]         
            # in_plus = input[...,None,:].repeat(1,1,d_size,1) + torch.eye(d_size, dtype=self.y0.dtype, device=self.y0.device)*eps #B x T x D          
            # in_minus = input[...,None,:].repeat(1,1,d_size,1) - torch.eye(d_size, dtype=self.y0.dtype, device=self.y0.device)*eps #B x T x D 
            # in_plus = input[None].repeat(d_size,1,1,1)  #D x B x T x D         
            # in_minus = input[None].repeat(d_size,1,1,1) #D x B x T x D 
            # for i in range(d_size):
            #     in_plus[i,:,:,i]+=eps
            #     in_minus[i,:,:,i]-=eps
            # jacob = ((self.func(t, in_plus.view(-1,t_size,d_size)) - self.func(t, in_minus.view(-1,t_size,d_size))) / 2*eps).reshape(d_size,*y0.shape)
            # pdb.set_trace()
            # trace = torch.diagonal(jacob.transpose(0,1).transpose(1,2), dim1=-2, dim2=-1).sum(-1)
            return trace            
        y0 = self.y0
        iterator = tqdm(zip(time_grid[:-1], time_grid[1:]), desc="ODE MDD")
        #iterator = zip(time_grid[:-1], time_grid[1:])
        ## from this version on, we only store the end-point jacobian and solution
        for i, (t0, t1) in enumerate(iterator):
            dt = t1 - t0  ## always greater than 0
            dy, f0 = self._step_func(self.func, t0, dt, None, y0)
            y1 = y0 + dy
            jacob_trace[i] = aabb_estimate(y1, t1)
            solution[i] = y1
            y0 = y1          
            
        return solution, jacob_trace

    def integrate_until_event(self, t0, event_fn):
        assert self.step_size is not None, "Event handling for fixed step solvers currently requires `step_size` to be provided in options."

        t0 = t0.type_as(self.y0.abs())
        y0 = self.y0
        dt = self.step_size

        sign0 = torch.sign(event_fn(t0, y0))
        max_itrs = 20000
        itr = 0
        while True:
            itr += 1
            t1 = t0 + dt
            dy, f0 = self._step_func(self.func, t0, dt, t1, y0)
            y1 = y0 + dy

            sign1 = torch.sign(event_fn(t1, y1))

            if sign0 != sign1:
                if self.interp == "linear":
                    interp_fn = lambda t: self._linear_interp(t0, t1, y0, y1, t)
                elif self.interp == "cubic":
                    f1 = self.func(t1, y1)
                    interp_fn = lambda t: self._cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t)
                else:
                    raise ValueError(f"Unknown interpolation method {self.interp}")
                event_time, y1 = find_event(interp_fn, sign0, t0, t1, event_fn, float(self.atol))
                break
            else:
                t0, y0 = t1, y1

            if itr >= max_itrs:
                raise RuntimeError(f"Reached maximum number of iterations {max_itrs}.")
        solution = torch.stack([self.y0, y1], dim=0)
        return event_time, solution

    def _cubic_hermite_interp(self, t0, y0, f0, t1, y1, f1, t):
        h = (t - t0) / (t1 - t0)
        h00 = (1 + 2 * h) * (1 - h) * (1 - h)
        h10 = h * (1 - h) * (1 - h)
        h01 = h * h * (3 - 2 * h)
        h11 = h * h * (h - 1)
        dt = (t1 - t0)
        return h00 * y0 + h10 * dt * f0 + h01 * y1 + h11 * dt * f1

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)  
    
class FixedGridODESolverHut(metaclass=abc.ABCMeta):
    order: int
    def __init__(self, func, y0, step_size=None, grid_constructor=None, interp="linear", perturb=False, **unused_kwargs):
        self.atol = unused_kwargs.pop('atol')
        unused_kwargs.pop('rtol', None)
        unused_kwargs.pop('norm', None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.dtype = y0.dtype
        self.device = y0.device
        self.step_size = step_size
        self.interp = interp
        self.perturb = perturb

        if step_size is None:
            if grid_constructor is None:
                self.grid_constructor = lambda f, y0, t: t
            else:
                self.grid_constructor = grid_constructor
        else:
            if grid_constructor is None:
                self.grid_constructor = self._grid_constructor_from_step_size(step_size)
            else:
                raise ValueError("step_size and grid_constructor are mutually exclusive arguments.")

    @classmethod
    def valid_callbacks(cls):
        return {'callback_step'}

    @staticmethod
    def _grid_constructor_from_step_size(step_size):
        def _grid_constructor(func, y0, t):
            start_time = t[0]
            end_time = t[-1]

            niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = torch.arange(0, niters, dtype=t.dtype, device=t.device) * step_size + start_time
            t_infer[-1] = t[-1]

            return t_infer
        return _grid_constructor

    @abc.abstractmethod
    def _step_func(self, func, t0, dt, t1, y0):
        pass

    def integrate(self, t, cond_mask, n_samples):
        time_grid = self.grid_constructor(self.func, self.y0, t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]
        ##XINWEI: MDD EXTRA assertion, no interpolation
        assert len(t) == len(time_grid)
        
        b_size = self.y0.shape[0]
        t_size = self.y0.shape[1]
        d_size = self.y0.shape[-1]

        solution = torch.zeros(len(t)-1, *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        ### HUT approx, return directly the phoneme-level jacob
        jacob_trace = torch.zeros(len(t)-1, *self.y0.shape[:-2], dtype=self.y0.dtype, device=self.y0.device)

        # with torch.enable_grad():
        y0 = self.y0
        #iterator = tqdm(zip(time_grid[1:-1], time_grid[2:]), desc="ODE MDD")
        iterator = zip(time_grid[:-1], time_grid[1:])
        
        def batch_v_p(v1, v2): ##for vmap, v1: P x D, v1_orig: N x P x D  return, 1 x 1
            return torch.bmm(v1.view(v1.shape[0], 1, -1), v2.view(v2.shape[0],-1,1)).sum()
        get_trace_batch = torch.vmap(batch_v_p, in_dims=(0,0))  #return N x 1
        ## from this version on, we only store the end-point jacobian and solution
        for i, (t0, t1) in enumerate(iterator):
            if i == 0: # no need grad
                dt = time_grid[1] - time_grid[0]  ## always greater than 0
                dy, f0 = self._step_func(self.func, t0, dt, None, y0)
            y1 = y0 + dy
            solution[i] = y1
            ##HUT approx
            ##step 1, sampling 
            v = torch.randint(0, 2, size=(b_size,n_samples,t_size,d_size), device=self.y0.device)*2 - 1 # B x N x T x D
            ##step 2, masking the targets
            v[cond_mask[:,None].expand(b_size,n_samples,t_size)] = 0
            ##step 3, consturct the input, enable the requried segment
            grad_list = []
            y1_list = []
            with torch.enable_grad():
                for b_idx in range(b_size):
                    target = y1[b_idx, ~cond_mask[b_idx]].requires_grad_(True)
                    left = y1[b_idx, 0:(~cond_mask[b_idx]).nonzero()[0]]
                    right = y1[b_idx, (~cond_mask[b_idx]).nonzero()[-1]+1:]
                    y1_row = torch.cat((left,target,right))
                    grad_list.append(target)
                    y1_list.append(y1_row)
                ##step 4, forward and backward
                dt = t1 - t0
                dy, f1 = self._step_func(self.func, t0, dt, None, torch.stack(y1_list))
                f1_list = [f1[b_idx] for b_idx in range(f1.shape[0])]  
                def get_trace_hut(v_in):
                    v_list = [v_in[b_idx] for b_idx in range(v_in.shape[0])]
                    vjp =torch.autograd.grad(f1_list, grad_list, v_list, retain_graph=True) ## B(tuple) x N x D Tensor
                    return vjp
                get_vjp_batch = torch.vmap(get_trace_hut, in_dims=1, out_dims=0)  
                vjp = get_vjp_batch(v)
                tr_list = [get_trace_batch(grad, v[idx, :, ~cond_mask[idx]].to(self.y0.dtype)).mean() for idx,grad in enumerate(vjp)]
                jacob_trace[i] = torch.tensor(tr_list, device=self.y0.device)              
                #del f1           
            y0 = y1.detach()
            #torch.cuda.empty_cache() 
            ##final steps              
            if i == time_grid.shape[0] - 3:
                y1 = y0 + dy
                solution[i+1] = y1
                ##HUT approx
                ##step 1, sampling 
                v = torch.randint(0, 2, size=(b_size,n_samples,t_size,d_size), device=self.y0.device)*2 - 1
                ##step 2, masking the targets
                v[cond_mask[:,None].expand(b_size,n_samples,t_size)] = 0
                ##step 3, consturct the input, enable the requried segment
                grad_list = []
                y1_list = []
                with torch.enable_grad():
                    for b_idx in range(b_size):
                        target = y1[b_idx, ~cond_mask[b_idx]].requires_grad_(True)
                        left = y1[b_idx, 0:(~cond_mask[b_idx]).nonzero()[0]]
                        right = y1[b_idx, (~cond_mask[b_idx]).nonzero()[-1]+1:]
                        y1_row = torch.cat((left,target,right))
                        grad_list.append(target)
                        y1_list.append(y1_row)
                    ##step 4, forward and backward
                    dt = 1
                    dy, f1 = self._step_func(self.func, t0, dt, None, torch.stack(y1_list))
                    f1_list = [f1[b_idx] for b_idx in range(f1.shape[0])]                 
                    def get_trace_hut(v_in):
                        v_list = [v_in[b_idx] for b_idx in range(v_in.shape[0])]
                        vjp =torch.autograd.grad(f1_list, grad_list, v_list, retain_graph=False) ## B(tuple) x N x D Tensor
                        return vjp
                    get_vjp_batch = torch.vmap(get_trace_hut, in_dims=1, out_dims=0)  
                    vjp = get_vjp_batch(v)
                    tr_list = [get_trace_batch(grad, v[idx, :, ~cond_mask[idx]].to(self.y0.dtype)).mean() for idx,grad in enumerate(vjp)]
                    jacob_trace[i+1] = torch.tensor(tr_list, device=self.y0.device)      
                    #slow version deprecated       
                    # def get_trace_hut(f1_in, y1_in, v_in, mask):
                    #     temp_grad = torch.autograd.grad(f1_in, y1_in, v_in, retain_graph=True)[0] ##  N x D Tensor
                    #     temp_trace = torch.bmm(temp_grad.view(temp_grad.shape[0], 1, -1), v_in[mask].view(v_in[mask].shape[0],-1,1).to(temp_grad.dtype)).sum()
                    #     return temp_trace
                    # get_trace_batch = torch.vmap(get_trace_hut, in_dims=(None, None, 0, None), out_dims=0)    
                    # for b_idx in range(b_size):
                    #     jacob_trace[i+1, b_idx] = get_trace_batch(f1[b_idx], grad_list[b_idx], v[b_idx], ~cond_mask[b_idx]).mean()     
                    #del f1
                #torch.cuda.empty_cache()                            
        return solution, jacob_trace

    def integrate_until_event(self, t0, event_fn):
        assert self.step_size is not None, "Event handling for fixed step solvers currently requires `step_size` to be provided in options."

        t0 = t0.type_as(self.y0.abs())
        y0 = self.y0
        dt = self.step_size

        sign0 = torch.sign(event_fn(t0, y0))
        max_itrs = 20000
        itr = 0
        while True:
            itr += 1
            t1 = t0 + dt
            dy, f0 = self._step_func(self.func, t0, dt, t1, y0)
            y1 = y0 + dy

            sign1 = torch.sign(event_fn(t1, y1))

            if sign0 != sign1:
                if self.interp == "linear":
                    interp_fn = lambda t: self._linear_interp(t0, t1, y0, y1, t)
                elif self.interp == "cubic":
                    f1 = self.func(t1, y1)
                    interp_fn = lambda t: self._cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t)
                else:
                    raise ValueError(f"Unknown interpolation method {self.interp}")
                event_time, y1 = find_event(interp_fn, sign0, t0, t1, event_fn, float(self.atol))
                break
            else:
                t0, y0 = t1, y1

            if itr >= max_itrs:
                raise RuntimeError(f"Reached maximum number of iterations {max_itrs}.")
        solution = torch.stack([self.y0, y1], dim=0)
        return event_time, solution

    def _cubic_hermite_interp(self, t0, y0, f0, t1, y1, f1, t):
        h = (t - t0) / (t1 - t0)
        h00 = (1 + 2 * h) * (1 - h) * (1 - h)
        h10 = h * (1 - h) * (1 - h)
        h01 = h * h * (3 - 2 * h)
        h11 = h * h * (h - 1)
        dt = (t1 - t0)
        return h00 * y0 + h10 * dt * f0 + h01 * y1 + h11 * dt * f1

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)  
   