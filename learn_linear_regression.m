
load D.dat

X = D(:,1:(end-1));
[m,d] = size(X);
X = [ones(m,1) X];
y = D(:,end);

w=inv(X'*X) * X' * y;
     
fid = fopen('w.dat','w');
fprintf(fid,'%.18e\n',w);
fclose(fid);

exit


