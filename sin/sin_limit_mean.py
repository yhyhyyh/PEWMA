from PEWMA.sin import sin_ARL_mean


def binary_search(M, hl, hu, A0, e1, e2,A, omega, theta0, rep, T, tau, shift, Lambda, p):
    global hp
    for i in range(M):
        hp = (hl + hu) / 2
        print('hp=',hp)
        ARL0 = sin_ARL_mean.calculate_ARL(A, omega, theta0, rep, T, tau, shift, Lambda, p, hp)
        print('ARL0=', ARL0)
        if (abs(ARL0 - A0)) < e1:
            print('The Accuracy Of ARL Reached')
            print('hp=', hp)
            break
        else:
            if ARL0 > A0:
                hu = hp
            if ARL0 < A0:
                hl = hp
                continue
        if abs(hu - hl) < e2:
            print('The Accuracy Of hp Reached')
            print('hp=', hp)
            break
        if i == M-1:
            print('The Estimation Accuracy Specified Cannot Be Reached')
    return hp
if __name__ == "__main__":
    print(binary_search(M=100, hl=0, hu=5, A0=200, e1=0.1, e2=0.0001, A=0.1, omega=0.5, theta0=1, rep=10000, T=1000, tau=50, shift=0, Lambda=0.1,p=5))
    # def binary_search(M, Ll, Lu, A0, e1, e2, rep,T,tau,theta,shift,S,q,p,m0,Lambda):


