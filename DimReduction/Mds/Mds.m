% colordef none;
% 
% n = 1000;
% [x y z] = build_swiss_roll(n);
% data = [x y z];
% 
% colors = get_cuadrant_colors(x,z);
% plot_swiss_roll(x,y,z, colors, [-15 15]);

n = 10;
p = 2;

H = eye(n) - ones(n)/n;
A = -0.5 * D.^2;
B = H * A * H;

[V, L] = eigs(B,p+3,'LR')
L_half = sqrt(L);
Y_full = V * L_half;
Y = Y(:,1:p)

plot(Y(:,1), Y(:,2), '.')
text(Y(:,1), Y(:,2), names)