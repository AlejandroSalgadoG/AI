rng(10); % for reproducibility
N = 2000;
t = 3*pi/2 * (1 + 2*rand(N,1));
h = 5 * rand(N,1);
X = [t.*cos(t), h, t.*sin(t)];
org = 2.5;

x = X(:,1);
y = X(:,2);
z = X(:,3);

C = 1:N;
C(x >= org & z > org) = 1;
C(x < org & z >= org) = 2;
C(x <= org & z < org) = 3;
C(x > org & z <= org) = 4;

figure
scatter3(x,y,z,36,C,'filled')
rotate3d on