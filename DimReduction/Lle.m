rng(10); 
n = 2000
data = build_swiss_roll(2000);
plot_swiss_roll(data, n, 2.5)

A = [
        [-1 -1];
        [-2 -1];
        [-3 -2];
        [1 1];
        [2 1];
        [3 2]
    ];

i= 1;

neighbors = k_neighbors(A, A(i,:), 2)