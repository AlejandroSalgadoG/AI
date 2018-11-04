rng(10); 
colordef white;
 
n = 1000;
[x y z] = build_swiss_roll(n);
data = [x y z];

colors = get_cuadrant_colors(x,z);
plot_swiss_roll(x,y,z, colors, [-15 15]);

k = 14;
p = 2;

dist_matrix = get_distance(data);

graph_matrix = zeros(n);

for i=1:n
    x_i = data(i,:);
    neighbors = k_neighbors(dist_matrix, i, k);
    graph_matrix(i, neighbors) = dist_matrix(i,neighbors);
    graph_matrix(neighbors,i) = dist_matrix(i,neighbors);
end

manifold = graph(graph_matrix);
D = distances(manifold);

H = eye(n) - ones(n)/n;
A = -0.5 * D.^2;
B = H * A * H;

[V, L] = eigs(B,p+3,'LR');
L_half = sqrt(L);
Y_full = V * L_half;
Y = Y_full(:,1:p);
size(colors)

plot_swiss_roll(Y(:,1),Y(:,1),zeros(n,1), colors, [-45 60])
