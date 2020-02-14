

def maxProfit(prices):
    start = 0
    flag = 0
    end = 0
    res = 0
    for i in range(len(prices) - 1):
        if prices[i] < prices[i + 1]:
            flag += 1
            #print(1)
        elif prices[i] > prices[i + 1] and i != 0:
            end = prices[i]
            res += (end - start)
            flag = 0
            #print(2)
        if flag == 1:
            start = prices[i]
            #print(3)
        if res == 0:
            res = prices[-1] - start
        # print(start)
        # print(end)
        # print(res)
    print(res)
if __name__ == '__main__':
    price = [1, 2, 3, 4, 5]
    maxProfit(price)