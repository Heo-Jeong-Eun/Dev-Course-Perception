import torch

# Quiz1
def quiz1():
    print("quiz1")
    a = torch.arange(1,7).view(2, 3)
    b = torch.arange(1,7).view(2, 3)
    
    sum = a + b
    sub = a - b
    
    a_sum = a.sum()
    b_sum = b.sum()
    
    print("Sum : {}".format(sum))
    print("Sub : {}".format(sub))
    print("a_sum : {} , b_sum {}".format(a_sum, b_sum))

# Quiz2
def quiz2():
    print("quiz2")
    a = torch.arange(1,46)
    a = a.view(1,5,3,3)
    b = torch.transpose(a, 1, 3)
    
    print("transposed : ", b[0,2,2,:])

# Quiz3
def quiz3():
    print("quiz3")
    a = torch.arange(1,7).view(2,3)
    b = torch.arange(1,7).view(2,3)
    
    c = torch.cat((a,b), dim = 1)
    
    d = torch.stack((a,b), dim = 0)
    
    print("concat {} \nstack {}". format(c, d))

if __name__ == "__main__":
    quiz1()
    quiz2()
    quiz3()