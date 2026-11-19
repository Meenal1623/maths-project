function StudentPerformanceAnalyzer(csvFile)
% STUDENTPERFORMANCEANALYZER  Analyze student marks & attendance and
% predict academic performance using regression.  (Base MATLAB, R2016b+)
%
%   StudentPerformanceAnalyzer            -> uses generated sample data
%   StudentPerformanceAnalyzer('x.csv')   -> uses your own CSV file
%
% CSV format (with header row):
%   Subject1,Subject2,...,SubjectK,Attendance,Group
%   (marks 0-100, attendance 0-100, group = 1 or 2)
%
% Techniques: matrices, descriptive statistics, correlation, scatter plots,
% simple linear regression, and hypothesis tests (slope t-test and
% two-sample Welch t-test).
%
% Outputs: console report, performance_report.txt, student_dashboard.png,
%          correlation_heatmap.png (and sample_students.csv if sample data).

    if nargin < 1, csvFile = ''; end
    close all; clc; rng(7);

    %% 1. LOAD / CREATE THE MARKS MATRIX
    [marks, attendance, group, subjects] = loadData(csvFile);
    [N, nSub] = size(marks);
    avgMarks  = mean(marks, 2);                 % exam performance per student
    fid = fopen('performance_report.txt', 'w');

    out(fid, '=====================================================\n');
    out(fid, '        STUDENT PERFORMANCE ANALYZER - REPORT\n');
    out(fid, '=====================================================\n');
    out(fid, 'Students: %d   Subjects: %s\n\n', N, strjoin(subjects, ', '));

    out(fid, '--- Marks matrix (first 8 students) ---\n');
    out(fid, '%-8s', 'ID');
    for j = 1:nSub, out(fid, '%-11s', subjects{j}); end
    out(fid, '%-11s%-8s\n', 'Attend%', 'Group');
    for i = 1:min(8, N)
        out(fid, '%-8d', i);
        for j = 1:nSub, out(fid, '%-11d', marks(i,j)); end
        out(fid, '%-11d%-8d\n', attendance(i), group(i));
    end
    out(fid, '\n');

    %% 2. DESCRIPTIVE STATISTICS
    meanV = mean(marks);   medianV = median(marks);   modeV = mode(marks);
    rangeV = max(marks) - min(marks);   varV = var(marks);   stdV = std(marks);

    out(fid, '--- Descriptive statistics ---\n');
    out(fid, '%-11s %8s %8s %8s %8s %10s %8s\n', ...
        'Subject','Mean','Median','Mode','Range','Variance','Std');
    for j = 1:nSub
        out(fid, '%-11s %8.2f %8.2f %8.0f %8.0f %10.2f %8.2f\n', subjects{j}, ...
            meanV(j), medianV(j), modeV(j), rangeV(j), varV(j), stdV(j));
    end
    out(fid, '%-11s %8.2f %8.2f %8.0f %8.0f %10.2f %8.2f\n\n', 'Average', ...
        mean(avgMarks), median(avgMarks), mode(round(avgMarks)), ...
        max(avgMarks)-min(avgMarks), var(avgMarks), std(avgMarks));

    %% 3. CORRELATION
    [R, P] = corrcoef(attendance, avgMarks);
    r = R(1,2);
    out(fid, '--- Correlation: attendance vs average marks ---\n');
    out(fid, 'Pearson r = %.4f (p = %.3g) -> %s\n', r, P(1,2), strength(r));
    for j = 1:nSub
        Rj = corrcoef(attendance, marks(:,j));
        out(fid, '  %-11s r = %.4f\n', subjects{j}, Rj(1,2));
    end
    out(fid, '\n');

    %% 4. LINEAR REGRESSION  (AvgMarks = b0 + b1*Attendance)
    idx = randperm(N);  nTr = round(0.8*N);
    tr = idx(1:nTr);    te = idx(nTr+1:end);

    Xtr = [ones(nTr,1), attendance(tr)];   ytr = avgMarks(tr);
    b   = Xtr \ ytr;
    res = ytr - Xtr*b;
    SSres = sum(res.^2);   SStot = sum((ytr - mean(ytr)).^2);
    R2  = 1 - SSres/SStot;

    % Hypothesis test on the slope:  H0: b1 = 0
    dfS  = nTr - 2;
    seB1 = sqrt((SSres/dfS) / sum((attendance(tr) - mean(attendance(tr))).^2));
    tB1  = b(2)/seB1;
    pB1  = betainc(dfS/(dfS + tB1^2), dfS/2, 0.5);

    Xte = [ones(numel(te),1), attendance(te)];
    yte = avgMarks(te);   ypr = Xte*b;
    RMSE = sqrt(mean((yte - ypr).^2));   MAE = mean(abs(yte - ypr));

    out(fid, '--- Linear regression ---\n');
    out(fid, 'AvgMarks = %.3f + %.3f * Attendance\n', b(1), b(2));
    out(fid, 'R^2 = %.4f | slope t = %.3f, p = %.3g (%s)\n', R2, tB1, pB1, ...
        ternary(pB1 < 0.05, 'attendance significantly predicts marks', 'slope not significant'));
    out(fid, 'Test RMSE = %.3f, MAE = %.3f (%d test students)\n\n', RMSE, MAE, numel(te));

    newAtt  = (50:10:100)';
    newPred = min(max([ones(numel(newAtt),1), newAtt]*b, 0), 100);
    out(fid, 'Predicted marks:\n  %-14s %-10s %s\n', 'Attendance(%)', 'Marks', 'Category');
    for k = 1:numel(newAtt)
        out(fid, '  %-14d %-10.2f %s\n', newAtt(k), newPred(k), category(newPred(k)));
    end
    out(fid, '\n');

    %% 5. HYPOTHESIS TEST: GROUP 1 vs GROUP 2 (Welch t-test)
    g1 = avgMarks(group == 1);   g2 = avgMarks(group == 2);
    alpha = 0.05;
    n1 = numel(g1); n2 = numel(g2);
    m1 = mean(g1);  m2 = mean(g2);  s1 = std(g1);  s2 = std(g2);
    se = sqrt(s1^2/n1 + s2^2/n2);
    tS = (m1 - m2)/se;
    df = (s1^2/n1 + s2^2/n2)^2 / ((s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1));
    pV = betainc(df/(df + tS^2), df/2, 0.5);
    tc = fzero(@(t) betainc(df/(df+t^2), df/2, 0.5) - alpha, [0.5 10]);
    ci = (m1 - m2) + [-1 1]*tc*se;

    out(fid, '--- Two-sample t-test (H0: equal group means) ---\n');
    out(fid, 'Group 1: n=%d mean=%.2f std=%.2f\n', n1, m1, s1);
    out(fid, 'Group 2: n=%d mean=%.2f std=%.2f\n', n2, m2, s2);
    out(fid, 't = %.4f, df = %.2f, p = %.3g, 95%% CI of difference = [%.2f, %.2f]\n', ...
        tS, df, pV, ci(1), ci(2));
    out(fid, 'Decision (alpha=0.05): %s\n\n', ...
        ternary(pV < alpha, 'REJECT H0 - groups differ significantly', ...
                            'FAIL TO REJECT H0 - no significant difference'));

    %% 6. PERFORMANCE REPORT SUMMARY
    cats = {'Distinction','First Class','Pass','Fail'};
    out(fid, '--- Performance categories ---\n');
    allCat = arrayfun(@(v) category(v), avgMarks, 'UniformOutput', false);
    for k = 1:numel(cats)
        c = sum(strcmp(allCat, cats{k}));
        out(fid, '  %-12s %3d students (%.1f%%)\n', cats{k}, c, 100*c/N);
    end
    [~, order] = sort(avgMarks, 'descend');
    out(fid, '\nTop 5 students:\n');
    for k = 1:min(5, N)
        out(fid, '  #%d Student %-3d avg=%.2f attendance=%d%%\n', k, order(k), ...
            avgMarks(order(k)), attendance(order(k)));
    end
    atRisk = find(avgMarks < 40);
    out(fid, '\nAt-risk students (avg < 40): %s\n', ...
        ternary(isempty(atRisk), 'none', mat2str(atRisk')));
    out(fid, '=====================================================\n');
    fclose(fid);

    %% GRAPHS
    xx = linspace(min(attendance), max(attendance), 100);

    f1 = figure('Name','Dashboard','Position',[50 50 1250 700]);
    subplot(2,3,1);
    bar(meanV); hold on; errorbar(1:nSub, meanV, stdV, 'k.', 'LineWidth', 1.2);
    set(gca,'XTick',1:nSub,'XTickLabel',subjects,'XTickLabelRotation',30);
    ylabel('Marks'); title('Mean marks +/- 1 SD'); grid on;

    subplot(2,3,2);
    histogram(avgMarks, 10); xlabel('Average marks'); ylabel('Students');
    title('Distribution of average marks'); grid on;

    subplot(2,3,3);
    scatter(attendance(tr), avgMarks(tr), 35, 'b', 'filled'); hold on;
    scatter(attendance(te), avgMarks(te), 45, 'r', 'filled');
    plot(xx, b(1) + b(2)*xx, 'k-', 'LineWidth', 2);
    xlabel('Attendance (%)'); ylabel('Average marks');
    title(sprintf('Scatter + regression (r=%.2f, R^2=%.2f)', r, R2));
    legend('Train','Test','Fit','Location','southeast'); grid on;

    subplot(2,3,4);
    scatter(yte, ypr, 50, 'filled'); hold on;
    lims = [min([yte;ypr])-3, max([yte;ypr])+3];
    plot(lims, lims, 'k--', 'LineWidth', 1.5); axis([lims lims]);
    xlabel('Actual'); ylabel('Predicted');
    title(sprintf('Actual vs predicted (RMSE=%.2f)', RMSE)); grid on;

    subplot(2,3,5);
    scatter(Xtr*b, res, 30, 'filled'); hold on; plot(xlim, [0 0], 'k--');
    xlabel('Fitted marks'); ylabel('Residual'); title('Residual plot'); grid on;

    subplot(2,3,6);
    bar([m1 m2]); hold on; errorbar(1:2, [m1 m2], [s1 s2], 'k.', 'LineWidth', 1.2);
    set(gca,'XTickLabel',{'Group 1','Group 2'}); ylabel('Mean average marks');
    title(sprintf('Welch t-test: t=%.2f, p=%.3g', tS, pV)); grid on;
    saveas(f1, 'student_dashboard.png');

    f2 = figure('Name','Correlation heatmap','Position',[100 100 600 520]);
    C = corrcoef([marks, attendance]);
    labels = [subjects, {'Attendance'}];
    imagesc(C, [-1 1]); colorbar; axis square;
    set(gca,'XTick',1:nSub+1,'XTickLabel',labels,'XTickLabelRotation',35, ...
            'YTick',1:nSub+1,'YTickLabel',labels);
    for i = 1:nSub+1
        for j = 1:nSub+1
            text(j, i, sprintf('%.2f', C(i,j)), 'HorizontalAlignment','center');
        end
    end
    title('Correlation matrix (marks & attendance)');
    saveas(f2, 'correlation_heatmap.png');

    fprintf('\nSaved: performance_report.txt, student_dashboard.png, correlation_heatmap.png\n');

    %% INTERACTIVE PREDICTION
    fprintf('\n--- Predict a student''s marks ---\n');
    while true
        a = input('Enter attendance % (press Enter to quit): ');
        if isempty(a), break; end
        if ~isscalar(a) || ~isnumeric(a) || a < 0 || a > 100
            fprintf('Please enter a number between 0 and 100.\n'); continue;
        end
        pm = min(max(b(1) + b(2)*a, 0), 100);
        fprintf('Attendance %.0f%% -> predicted marks %.2f (%s)\n', a, pm, category(pm));
    end
end

%% ============================ LOCAL FUNCTIONS ===========================
function [marks, attendance, group, subjects] = loadData(csvFile)
    if isempty(csvFile)
        N = 60;
        subjects = {'Maths','Physics','Chemistry','English','Computer'};
        group = [ones(30,1); 2*ones(30,1)];
        attendance = [86 + 6*randn(30,1); 72 + 10*randn(30,1)];
        attendance = min(max(round(attendance), 40), 100);
        marks = zeros(N, numel(subjects));
        for j = 1:numel(subjects)
            marks(:,j) = 0.65*attendance + 12 + 6*randn(N,1) + 2*(j-3);
        end
        marks = min(max(round(marks), 0), 100);
        T = array2table([marks, attendance, group], ...
            'VariableNames', [subjects, {'Attendance','Group'}]);
        writetable(T, 'sample_students.csv');
        fprintf('Using generated sample data (saved as sample_students.csv)\n');
    else
        T = readtable(csvFile);
        vn = T.Properties.VariableNames;
        A  = table2array(T);
        marks      = A(:, 1:end-2);
        attendance = A(:, end-1);
        group      = A(:, end);
        subjects   = vn(1:end-2);
        fprintf('Loaded %s\n', csvFile);
    end
end

function out(fid, fmt, varargin)       % print to screen AND report file
    fprintf(1,   fmt, varargin{:});
    fprintf(fid, fmt, varargin{:});
end

function s = strength(r)
    a = abs(r);
    names = {'very weak/none','weak','moderate','strong','very strong'};
    s = names{1 + (a>=0.2) + (a>=0.4) + (a>=0.6) + (a>=0.8)};
    if r >= 0, s = [s ' positive']; else, s = [s ' negative']; end
end

function c = category(m)
    if     m >= 75, c = 'Distinction';
    elseif m >= 60, c = 'First Class';
    elseif m >= 40, c = 'Pass';
    else,           c = 'Fail';
    end
end

function v = ternary(cond, a, b)
    if cond, v = a; else, v = b; end
end