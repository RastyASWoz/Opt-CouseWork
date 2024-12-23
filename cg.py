import numpy as np

def CG(A, b, x0, tol=1e-10, max_iter=1000):
    r = b - A @ x0
    p = r.copy()
    rs_old = r.T @ r

    x = x0.copy()
    for i in range(max_iter):
        alpha = rs_old /( p.T @ A @ p)
        x += alpha * p
        r -= alpha * A@p
        rs_new = r.T @ r
        
        if np.sqrt(rs_new) < tol:
            print(f'次数: {i+1} 残差: {np.sqrt(rs_new)}')
            break
        
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    
    return x

def f_punish(A,b,C,t,lam):
    EA = 2*(A+lam*C.T@C)
    Eb = b+2*lam*t.T@C
    #print(EA,Eb)
    r_EA = np.linalg.inv(EA)
    #print(r_EA)
    print(r_EA@Eb)
    return EA,Eb

def reslove(A,b,C,t,lam):
    x0 = np.zeros(A.shape[0])
    A,b = f_punish(A,b,C,t,lam)
    return CG(A,b,x0)

if __name__ == "__main__":
    # 示例一：
    # 目标： min x^2 + y^2 - 6x - 4y
    # 约束： x + y = 4
    # 解析解： x = 2.5 , y = 1.5
    
    A = np.array([[1, 0], [0, 1]])
    b = np.array([6, 4])
    C = np.array([[1,1]])
    t = np.array([4])
    solution = reslove(A, b, C, t, 100)
    print("Solution:", solution)
    # 运行结果： [2.50248756 1.50248756]
    
    # 示例二：
    # 目标： min 4x^2 + 3y^2 + 6z^2 + t^2 - 2x - 3y - 4z + 5t - 3xy + 2xz - 4yt
    # 约束： x + y + z + t = 10, x + t = 5
    A = np.array([[4, -1.5, 1, 0], 
                [-1.5, 3, 0, 0], 
                [1, 0, 6, -2], 
                [0, 0, -2, 1]])
    b = np.array([2, 3, 4, -5])
    C = np.array([[1, 1, 1, 1], [1, 0, 0, 1]])
    t = np.array([10, 5])
    solution = reslove(A, b, C, t, 100)
    print("Solution:", solution)
    # 运行结果： [1.18181818 2.69941223 2.22309003 3.84742447]
