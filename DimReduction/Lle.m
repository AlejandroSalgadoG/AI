rng(10); 
n = 2000;
[x, y, z] = build_swiss_roll(n);
data = [x y z];

%colors = get_cuadrant_colors(x,z,2.5);
%plot_swiss_roll(x,y,z, colors)

i= 17;

k = 3;

x_i = data(i,:);
neighbors = k_neighbors(data, x_i, k);

v_i = data(neighbors,:)';
e = ones(k,1);
x_e = repmat(x_i, k, 1)';
 
g = (x_e - v_i)' * (x_e - v_i);
w_i = g\e;
w_i = w_i / sum(w_i);

norm( v_i*w_i - x_i')

colors = paint_neighbors(n, i, neighbors); 
% mask = x > -10 & x < -9 & z > -1;
% plot_swiss_roll(x(mask),y(mask),z(mask), colors(mask))
plot_swiss_roll(x,y,z, colors)