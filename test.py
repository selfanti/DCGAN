def hello():
    print("step 1")
    yield 1
    print("step 2")
    yield 2
    print("step 3")
    yield 3
    #返回值为生成器

g=hello()
print((next(g)))