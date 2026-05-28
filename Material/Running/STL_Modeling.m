n = 3;
r1 = zeros(n^3, 3);
for a = 1:n
    for b = 1:n
        for c = 1:n
            r1(n*n*a - n*n + n*b - n + c - 1 + 1, :) = [a, b, c];
        end
    end
end
r2_data = readtable('Porosity.xlsx');
r2 = r2_data.Var1;

if numel(r2) ~= n^3
    warning('r2 数据的数量不匹配 n^3。请检查数据文件。');
end

r2 = -2.8218 * r2 + 2.352;
data0 = [r1, r2];
b = 3;
sizeofdata0 = [n, n, n];
accu = 21;
v = createv_2(data0, sizeofdata0, accu, b);
sizeofgyroid = 2;
stldataofgyroid = drawgyroid(v, sizeofdata0, accu, sizeofgyroid, sizeofgyroid * data0 - 1);
name = ['1.stl'];
stlwrite(1, name, stldataofgyroid);