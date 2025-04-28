import random
print('whats your name ?')

trials=0
max_trials=10
xMin=-50
xMax=50
number=random.randint(xMin,xMax)
while trials<max_trials:
    trials+=1
    guess=int(input("enter guess between {} to {}".format(xMin,xMax)))
    if guess==number:
        print('Congrates!')
        break
    else:
        if guess<number:
            print('too low')
        else:
            print("too high")
        print('U have{} more tries'.format(max_trials-trials))
#
# x=random.randint(-19,25)
# print(x)