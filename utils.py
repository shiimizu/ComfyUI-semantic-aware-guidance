import torch
import numbers
import math
import torch.nn.functional as F

class GaussianSmoothing(torch.nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(
            input,
            weight=self.weight.to(device=input.device, dtype=input.dtype),
            groups=self.groups,
        )

def ceildiv(a,b):
    return -(a//-b)

def getshape(tx, tt):
    return [tx]+ [1] * (len(tt.shape) - 1)

def rep(x:torch.Tensor, n:int=None) -> torch.Tensor:
    ''' repeat the tensor on its first dim '''
    if isinstance(n,bool) and n == 1: return x
    B = x.shape[0]
    n2=n
    if isinstance(n, torch.Tensor):
        n = ceildiv(n.shape[0], B)
    r_dims = len(x.shape) - 1
    if B == 1:      # batch_size = 1 (not `tile_batch_size`)
        shape = [n] + [-1] * r_dims     # [N, -1, ...]
        return x.expand(shape)          # `expand` is much lighter than `tile`
    else:
        shape = [n] + [1] * r_dims      # [N, 1, ...]
        out = x.repeat(shape)
        if isinstance(n2, torch.Tensor):
            out = out[:n2.shape[0]]
        else:
            out = out[:n]
        return out

# from comfy/ldm/modules/attention.py
# but more efficient and modified to return attention scores as well as output. unused.
def attention_basic_with_sim(q, k, v, heads, mask=None, attn_precision=None, out=True):
    b, _, dim_head = q.shape
    dim_head //= heads
    scale = dim_head ** -0.5

    h = heads
    q, k, v = map(
        lambda t: t.unsqueeze(3)
        .reshape(b, -1, heads, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b * heads, -1, dim_head)
        .contiguous(),
        (q, k, v),
    )

    # force cast to fp32 to avoid overflowing
    if attn_precision == torch.float32:
        # sim = einsum('b i d, b j d -> b i j', q.float(), k.float()) * scale
        sim = torch.bmm(q.float(), k.float().transpose(-1, -2)) * scale
    else:
        # sim = einsum('b i d, b j d -> b i j', q, k) * scale
        sim = torch.bmm(q, k.transpose(-1, -2)) * scale

    del q, k

    if mask is not None:
        # mask = rearrange(mask, 'b ... -> b (...)')
        mb, mh = mask.shape[:2]
        mask = mask.view(b, -1)
        max_neg_value = -torch.finfo(sim.dtype).max
        # mask = repeat(mask, 'b j -> (b h) () j', h=h)
        mask = mask.unsqueeze_(1).repeat(mh, 1, 1).view(mb * mh, 1, -1)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    sim = sim.softmax(dim=-1)

    if not out: return (None, sim)
    # out = einsum('b i j, b j d -> b i d', sim.to(v.dtype), v)
    out = torch.bmm(sim.to(v.dtype), v)
    out = (
        out.unsqueeze_(0)
        .reshape(b, heads, -1, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b, -1, heads * dim_head)
    )
    return (out, sim)
