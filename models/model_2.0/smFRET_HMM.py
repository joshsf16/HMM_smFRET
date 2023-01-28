import torch
from tqdm import tqdm

class PhotonDataset(torch.utils.data.Dataset):
  def __init__(self, lines):
    self.lines = lines # list of observations of length 3
    collate = Collate() # function for generating a minibatch from strings
    self.loader = torch.utils.data.DataLoader(self, batch_size=1024, num_workers=0, shuffle=True, collate_fn=collate)

  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    return line

class Collate:
  def __init__(self):
    pass

  def __call__(self, batch):
    """
    Returns a minibatch of strings, padded to have the same length.
    """
    x = []
    batch_size = len(batch)
    for index in range(batch_size):
      x.append(batch[index])

    # pad all sequences with 0 to have same length
    x_lengths = [len(x_) for x_ in x]
    T = max(x_lengths)
    for index in range(batch_size):
      x[index] += [0] * (T - len(x[index]))
      x[index] = torch.tensor(x[index])

    # stack into single tensor
    x = torch.stack(x)
    x_lengths = torch.tensor(x_lengths)
    return (x,x_lengths)

class Trainer:
    """
    Trainer for HMM model.
    """
    def __init__(self, model, lr):
        """
        Initializes a new Trainer.

        Attributes:
        model (HMM): HMM model to be trained.
        lr (float): learning rate for torch.optim.Adam algorithm.
        """
        # Use Adam optimizer algorithm.
        self.model = model
        self.lr = lr
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.00001)
  
    def train(self, dataset):
        train_loss = 0
        num_samples = 0
        self.model.train()
        print_interval = 50
        for idx, batch in enumerate(tqdm(dataset.loader)):
            x,T = batch
        batch_size = len(x)
        num_samples += batch_size
        log_probs = self.model(x,T)

        loss = -log_probs.mean() 
        # this step is key. we want to maximize the log probablity, or minimize the negative log probability mean of the forward algorithm.

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        train_loss += loss.cpu().data.numpy().item() * batch_size
        if idx % print_interval == 0:
            print("loss:", loss.item())
            for _ in range(5):
                sampled_x, sampled_z = self.model.sample()
                print(sampled_x)
                print(sampled_z)
        train_loss /= num_samples
        return train_loss

    def test(self, dataset):
        test_loss = 0
        num_samples = 0
        self.model.eval()
        print_interval = 50
        for idx, batch in enumerate(dataset.loader):
            x,T = batch
        batch_size = len(x)
        num_samples += batch_size
        log_probs = self.model(x,T)
        loss = -log_probs.mean()
        test_loss += loss.cpu().data.numpy().item() * batch_size
        if idx % print_interval == 0:
            print("loss:", loss.item())
            sampled_x, sampled_z = self.model.sample()
            print(sampled_x)
            print(sampled_z)
        test_loss /= num_samples
        return test_loss

class HMM(torch.nn.Module): # torch documentation suggests inheritance from torch.nn.Module
    """
    Hidden Markov Model with discrete observations.
    """
    def __init__(self, transitions, emissions, priors):
        """
        Initializes a new HMM model.

        NOTE: The variables 'transitions', 'emissions', and 'priors' should be of type lists. 
        They will be normalized using torch.nn.functional softmax functions.

        Attributes:
        N (int): the number of states
        M (int): the number of observations
        transition_model (TransitionModel): the transition matrix for this HMM
        emission_model (EissionModel): the emission matrix for this HMM
        state_priors (torch.nn.Parameter): the prior distribution for this HMM
        is_cuda (bool): if a GPU is activated using cuda() for use
        """
        super().__init__()

        # First, save the number of observations and the number of states
        self.N = len(priors) # number of states
        self.M = len(emissions[0]) # number of observations

        # For the purposes of sampling and other algos, we will keep inputted probabilities unnormalized and pre-process data as needed.

        # Create A
        self.unnormalized_trans = TransitionMatrix(self.N, transitions)

        # b(x_t)
        self.unnormalized_emiss = EmissionMatrix(self.N, self.M, emissions)

        # pi
        self.unnormalized_sp = torch.nn.Parameter(torch.Tensor(priors))
    
        # use the GPU, for speed
        if torch.cuda.is_available(): 
            self.cuda()
            self.is_cuda = True

        else: self.is_cuda = False

    def sample(self, T=10):
        """
        This function samples the HMM model, returning the hidden states and what was observable.
        
        This function also locally normalizes the unnormalized_sp, unnormalized_trans, unnormalized_emiss 
        using the torch.nn.functional.softmax(...)
        """
        state_priors = torch.nn.functional.softmax(self.unnormalized_sp, dim=0)
        emission_matrix = torch.nn.functional.softmax(self.unnormalized_emiss.matrix, dim=1)
        transition_matrix = torch.nn.functional.softmax(self.unnormalized_trans.matrix, dim=0)

        # sample initial state
        z_t = torch.distributions.categorical.Categorical(state_priors).sample().item()
        z = []
        x = []
        z.append(z_t)
        
        for t in range(0,T):
            # sample emission
            x_t = torch.distributions.categorical.Categorical(emission_matrix[z_t]).sample().item()
            x.append(x_t)

            # sample transition
            z_t = torch.distributions.categorical.Categorical(transition_matrix[:,z_t]).sample().item()
            if t < T-1: z.append(z_t)

        return x, z

    def forward(self, x, T):
        """
        x : IntTensor of shape (batch size, T_max)
        T : IntTensor of shape (batch size)

        Compute log p(x) for each example in the batch.
        T = length of each example 

        This function also locally normalizes the unnormalized_sp, unnormalized_trans, unnormalized_emiss 
        using the torch.nn.functional.log_softmax(...)

        Worth noting, batch size is just the number of observation <<sequences>> passed to the forward algorithm for probability calculation.
        """
        if self.is_cuda:
            x = x.cuda()
            T = T.cuda()

        batch_size = x.shape[0] # how many sequences we'll be calculating for
        T_max = x.shape[1] # the number of time observations

        log_state_priors = torch.nn.functional.log_softmax(self.unnormalized_sp, dim=0) # this log normalizes state priors
        log_alpha = torch.zeros(batch_size, T_max, self.N) # creates alpha prob matrix in R3. firs dim is batch or sequence number, second is time observation, and last is states
        
        if self.is_cuda: 
            log_alpha = log_alpha.cuda()

        # SPECIAL NOTE: self.unnormalized_emiss(x[:,0]) will invoke the function __call__(...) from EmissionMatrix that then implicitly calls emission_model_forward(self, x[:,0])
            
        log_alpha[:, 0, :] = self.unnormalized_emiss(x[:,0]) + log_state_priors
        for t in range(1, T_max):
            log_alpha[:, t, :] = self.unnormalized_emiss(x[:,t]) + self.unnormalized_trans(log_alpha[:, t-1, :])

        # Select the sum for the final timestep (each x may have different length).
        log_sums = log_alpha.logsumexp(dim=2)
        log_probs = torch.gather(log_sums, 1, T.view(-1,1) - 1)

        return log_probs.exp()

    def viterbi(self, x, T):
        """
        x : IntTensor of shape (batch size, T_max)
        T : IntTensor of shape (batch size)
        Find argmax_z log p(x|z) for each (x) in the batch.
        """
        if self.is_cuda:
            x = x.cuda()
            T = T.cuda()

        batch_size = x.shape[0] 
        T_max = x.shape[1]

        log_state_priors = torch.nn.functional.log_softmax(self.unnormalized_sp, dim=0)
        log_delta = torch.zeros(batch_size, T_max, self.N).float()
        psi = torch.zeros(batch_size, T_max, self.N).long()
        
        if self.is_cuda:
            log_delta = log_delta.cuda()
            psi = psi.cuda()

        log_delta[:, 0, :] = self.unnormalized_emiss(x[:,0]) + log_state_priors
        for t in range(1, T_max):
            max_val, argmax_val = self.unnormalized_trans.maxmul(log_delta[:, t-1, :])
            log_delta[:, t, :] = self.unnormalized_emiss(x[:,t]) + max_val
            psi[:, t, :] = argmax_val

        # Get the log probability of the best path
        log_max = log_delta.max(dim=2)[0]
        best_path_scores = torch.gather(log_max, 1, T.view(-1,1) - 1)

        # This next part is a bit tricky to parallelize across the batch,
        # so we will do it separately for each example.
        z_star = []
        for i in range(0, batch_size):
            z_star_i = [ log_delta[i, T[i] - 1, :].max(dim=0)[1].item() ]
            for t in range(T[i] - 1, 0, -1):
                z_t = psi[i, t, z_star_i[0]].item()
                z_star_i.insert(0, z_t)

            z_star.append(z_star_i)

        return z_star, best_path_scores # return both the best path and its log probability

class TransitionMatrix(torch.nn.Module):
    """
    The transition matrix for our HMM model.
    """
    def __init__(self, N, transitions):
        """
        Instantiates a new transition matrix for our HMM model.
        """
        ### Checks to make sure that the number of priors and transitions line up
        if len(transitions) != N:
            raise ValueError(f'Mismatch in the number of priors and rows/cols in "transitions". {N} != {len(transitions)}')

        super().__init__()
        self.N = N
        self.matrix = torch.nn.Parameter(torch.Tensor(transitions))

    def forward(self, log_alpha):
        """
        log_alpha : Tensor of shape (batch size, N)
        Multiply previous timestep's alphas by transition matrix (in log domain)
        """
        log_transition_matrix = torch.nn.functional.log_softmax(self.matrix, dim=0)

        # Matrix multiplication in the log domain
        out = log_domain_matmul(log_transition_matrix, log_alpha.transpose(0,1)).transpose(0,1)
        return out
    
    def transition_model_maxmul(self, log_alpha):
        log_transition_matrix = torch.nn.functional.log_softmax(self.matrix, dim=0)

        out1, out2 = maxmul(log_transition_matrix, log_alpha.transpose(0,1))
        return out1.transpose(0,1), out2.transpose(0,1)

class EmissionMatrix(torch.nn.Module):
    def __init__(self, N, M, emissions):
        """
        Instantiates a new emission matrix for our HMM model.
        """
        ### Checks if the number of states and rows of the emissions matrix line up
        if len(emissions) != N:
            raise ValueError(f'Mismatch in the number of priors and rows in "emissions". {N} != {len(emissions)}')

        super().__init__()
        self.N = N
        self.M = M
        self.matrix = torch.nn.Parameter(torch.Tensor(emissions))
    
    def forward(self, x_t):
        log_emission_matrix = torch.nn.functional.log_softmax(self.matrix, dim=1)
        out = log_emission_matrix[:, x_t].transpose(0,1)
        return out

def log_domain_matmul(log_A, log_B):
	"""
	log_A : m x n
	log_B : n x p
	output : m x p matrix

	Normally, a matrix multiplication
	computes out_{i,j} = sum_k A_{i,k} x B_{k,j}

	A log domain matrix multiplication
	computes out_{i,j} = logsumexp_k log_A_{i,k} + log_B_{k,j}
	"""
	m = log_A.shape[0]
	n = log_A.shape[1]
	p = log_B.shape[1]

	log_A_expanded = torch.reshape(log_A, (m,n,1))
	log_B_expanded = torch.reshape(log_B, (1,n,p))

	elementwise_sum = log_A_expanded + log_B_expanded
	out = torch.logsumexp(elementwise_sum, dim=1)

	return out

def maxmul(log_A, log_B):
	"""
	log_A : m x n
	log_B : n x p
	output : m x p matrix

	Similar to the log domain matrix multiplication,
	this computes out_{i,j} = max_k log_A_{i,k} + log_B_{k,j}
	"""
	m = log_A.shape[0]
	n = log_A.shape[1]
	p = log_B.shape[1]

	log_A_expanded = torch.stack([log_A] * p, dim=2)
	log_B_expanded = torch.stack([log_B] * m, dim=0)

	elementwise_sum = log_A_expanded + log_B_expanded
	out1,out2 = torch.max(elementwise_sum, dim=1)

	return out1,out2