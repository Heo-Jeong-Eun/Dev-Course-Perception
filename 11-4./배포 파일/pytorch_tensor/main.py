import torch
import numpy as np

def make_tensor():
    # int
    a = torch.tensor([1], dtype=torch.int16)
    # float
    b = torch.tensor([1], dtype=torch.float32)
    # double
    c = torch.tensor([1], dtype=torch.float64)
    print(a, b, c)
    tensor_list = [a, b, c]
    
    for t in tensor_list:
        print("shape of a tensor {}".format(t.shape))
        print("datatype of a tensor {}".format(t.dtype))
        print("device a tensor is stored on {}".format(t.device))
        print("-----")

    # zeros
    d = torch.zeros(4)
    # ones
    e = torch.ones(4)
    # random
    f = torch.rand(4)
    print("zeros : {} \n ones : {} \n random : {}".format(d, e, f))

def sumsub_tensor():
    a = torch.tensor([3,2])
    b = torch.tensor([5,3])
    print("input a : {} \n input b : {}".format(a, b))
    # sum
    sum = a + b
    print("sum : {}".format(sum))

    # sub
    sub = a - b
    print("sub : {}".format(sub))
    
    sum_all_a, sum_all_b = a.sum(), b.sum()
    print("sum")

def mul_div():
    a = torch.arange(0,9).view(3,3)
    b = torch.arange(0,9).view(3,3)

    print("input a : {} \n input b : {}".format(a, b))
    
    # matrix_multiplication
    c = torch.matmul(a,b)

    # elementwise multiplication
    d = torch.mul(a,b)
    
    print("multiplication : {}".format(c))
    print("elementwise multiplication : {}".format(d))

def reshape_tensor():
    a = torch.tensor([2.5,5.6,9.1,4.6,3.2,6.5])

    # reshape 1d to 2d
    b = a.view(2,3)
    print("1D tensor : {} -> reshaped 2D tensor : {}".format(a, b))
    
    # transpose
    b_t = b.t()

    c = torch.randn(2,2)
    print("rand tensor : {}".format(c))

    d1 = torch.arange(0,8)
    d2 = d1.view(2,4)
    print("1D tensor : {} -> reshaped 2D tensor : {}".format(d1, d2))

def access_tensor():
    a = torch.arange(1,13).view(4,3)

    print("tensor : {}".format(a))

    # first column (slicing)
    b = a[:,0]
    # first row (slicing)
    c = a[0,:]
    # [1,1] (indexing)
    d = a[1,1]

    print("first column : {}, \n first row {}, \n a[1,1] {}".format(b,c,d))

def transform_numpy():
    a = torch.arange(1,13).view(4,3)
    print("tensor : {}".format(a))

    a_np = a.numpy()

    print("tensor : {} => numpy : {}".format(a, a_np))

    b = np.array([1,2,4])

    b_tensor = torch.from_numpy(b)

    print("tensor : {} => numpy : {}".format(b, b_tensor))

def concat_tensor():
    a = torch.arange(1,10).view(3,3)
    b = torch.arange(10,19).view(3,3)
    c = torch.arange(19,28).view(3,3)

    abc = torch.cat([a,b,c],dim=1)

    print(a, b, c)
    print("concat : {}".format(abc))

def stack_tensor():
    a = torch.arange(1,10).view(3,3)
    b = torch.arange(10,19).view(3,3)
    c = torch.arange(19,28).view(3,3)

    abc = torch.stack([a,b,c],dim=0)

    print(a, b, c)
    print("stack : {}".format(abc))

def transpose_tensor():
    a = torch.arange(1,10).view(3,3)
    print("a tensor : {}".format(a))
    
    a_t = torch.transpose(a,0,1)
    
    print("a transposed : {}".format(a_t))
    
    b = torch.arange(1,25).view(4,3,2)
    
    print("b tensor : {}".format(b))
    
    b_t = torch.transpose(b,0,2)
    print("b transposed : {}".format(b_t))
    
    b_p = b.permute(2,0,1)
    print("b permute : {}".format(b_p))
    
if __name__ == "__main__":
    make_tensor()
    sumsub_tensor()
    reshape_tensor()
    mul_div()
    access_tensor()
    transform_numpy()
    concat_tensor()
    stack_tensor()
    transpose_tensor()