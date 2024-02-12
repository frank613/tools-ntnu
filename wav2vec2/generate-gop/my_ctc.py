import torch
import numpy as np

def compute_gradient(params, seq, alphas, betas):
    pass

class MyCTC(torch.Function):
    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(params, seq, blank=0):
        """
        CTC loss function.
        params - n x m matrix of n-D probability distributions(softmax output) over m frames.
        seq - sequence of phone id's for given example.
        Returns objective, alphas and betas.
        """
        seqLen = seq.shape[0] # Length of label sequence (# phones)
        numphones = params.shape[0] # Number of labels
        L = 2*seqLen + 1 # Length of label sequence with blanks
        T = params.shape[1] # Length of utterance (time)

        alphas = torch.zeros((L,T)).double()
        betas = torch.zeros((L,T)).double()

            # Initialize alphas and forward pass 
        alphas[0,0] = params[blank,0]
        alphas[1,0] = params[seq[0],0]

        for t in range(1,T):
            start = max(0,L-2*(T-t)) 
            end = min(2*t+2,L)
            for s in range(start,L):
                l = int((s-1)/2)
                # blank
                if s%2 == 0:
                    if s==0:
                        alphas[s,t] = alphas[s,t-1] * params[blank,t]
                    else:
                        alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[blank,t]
                # same label twice
                elif s == 1 or seq[l] == seq[l-1]:
                    alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[seq[l],t]
                else:
                    alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1] + alphas[s-2,t-1]) \
                        * params[seq[l],t]
            
        llForward = torch.log(alphas[L-1, T-1] + alphas[L-2, T-1])

        # Initialize betas and backwards pass
        betas[-1,-1] = params[blank,-1]
        betas[-2,-1] = params[seq[-1],-1]
        c = betas[:,-1].sum()
        betas[:,-1] = betas[:,-1]
        for t in range(T-2,-1,-1):
            start = max(0,L-2*(T-t)) 
            end = min(2*t+2,L)
            for s in range(end-1,-1,-1):
                l = int((s-1)/2)
                # blank
                if s%2 == 0:
                    if s == L-1:
                        betas[s,t] = betas[s,t+1] * params[blank,t]
                    else:
                        betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * params[blank,t]
                # same label twice
                elif s == L-2 or seq[l] == seq[l+1]:
                    betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * params[seq[l],t]
                else:
                    betas[s,t] = (betas[s,t+1] + betas[s+1,t+1] + betas[s+2,t+1]) \
                        * params[seq[l],t]

        llBackward = torch.log(betas[0, 0] + betas[1, 0])
        # Check for underflow or zeros in denominator of gradient
        llDiff = np.abs(llForward-llBackward)
        if llDiff > 1e-5 :
            print("Diff in forward/backward LL : %f"%llDiff)
        
        return -llForward,alphas,betas

@staticmethod
# inputs is a Tuple of all of the inputs passed to forward.
# output is the output of the forward().
def setup_context(ctx, inputs, output):
    params, seq = inputs
    _, alphas, betas = output
    ctx.save_for_backward(params, seq, alphas, betas)

# This function has only a single output, so it gets only one gradient
@staticmethod
def backward(ctx, grad_output):
    # This is a pattern that is very convenient - at the top of backward
    # unpack saved_tensors and initialize all gradients w.r.t. inputs to
    # None. Thanks to the fact that additional trailing Nones are
    # ignored, the return statement is simple even when the function has
    # optional inputs.
    params, seq, alphas, betas = ctx.saved_tensors
    grad_params = compute_gradient(params, seq, alphas, betas) 
    
    return (grad_params, None)


def train():
    seq = torch.rand(10)
    logits = torch.rand(20,30)
    myctc = MyCTC.apply
    likelihood, _, _ = myctc(seq,logits)
    likelihood.backward(1,1,1)

