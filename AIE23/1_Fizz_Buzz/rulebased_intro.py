def func(max):
    res = []
    for i in range(1, max):
        # 对15取余为0 输出fizzbuzz
        if i % 15 == 0:
            res.append('fizzbuzz')
        # 对3取余为0，输出fizz
        elif i % 3 == 0:
            res.append('fizz')
        # 对5取余为0，输出为buzz
        elif i % 5 == 0:
            res.append('buzz')
        # 不符合以上3种情况，直接输出数字
        else:
            res.append(str(i))
    print(' '.join(res))

func(max=100)
