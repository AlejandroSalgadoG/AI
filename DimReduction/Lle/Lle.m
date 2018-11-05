rng(10); 
colordef white;

n = 1000;
[x, y, z] = build_swiss_roll(n);
data = [x y z];

colors = get_cuadrant_colors(x,z);
plot_swiss_roll(x,y,z, colors, [-15 15])

k = 12;
p = 2;

tol=0.001;

W = zeros(n);
e = ones(k,1);
for i=1:n
    x_i = data(i,:);
    neighbors = k_neighbors(data, x_i, k);
    v_i = data(neighbors,:)';
    x_e = repmat(x_i, k, 1)';
    g = (x_e - v_i)' * (x_e - v_i);
    g = g + eye(k)*tol*trace(g); % regularization
    w_i = g\e;
    w_i = w_i / sum(w_i);
    %norm( v_i*w_i - x_i')
    W(neighbors,i) = w_i;
end

% data_aprox = W' * data;
% plot_swiss_roll(data_aprox(:,1),data_aprox(:,2),data_aprox(:,3), colors, [-15 15])

I = eye(n);
M = (I-W) * (I-W)';
[eigvec, ~] = eigs(M,p+1,'SM');
Y = eigvec(:,2:p+1);

plot_swiss_roll(Y(:,1),Y(:,1),zeros(n,1), colors, [-0.1 0.1])