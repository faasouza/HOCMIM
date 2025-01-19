function [S, tElapsed, CMIapp, par, Order] = HOCMIM(X, Y, K, n, fast, verbose)
    % Parallel implementation of HOCMIM feature selection
    % Author: Francisco Alexandre
    % Date: 11/2020, Version: 1.0

    % Check input arguments
    if nargin < 6
        verbose = false;
    end

    % Initialize variables
    N = size(X, 2);
    if K > N
        K = N;
    end
    tStart = tic; % Start timer
    A = 1:N; % Feature indices
    S = zeros(1, K); % Selected features
    CMIapp = zeros(1, K); % Conditional mutual information approximation

    % Compute mutual information for the first feature
    MI = arrayfun(@(i) mutual_info(X(:, i), Y), 1:N);
    [max_MI, max_idx] = max(MI);

    % Initialize selected features and order
    S(1) = A(max_idx);
    CMIapp(1) = max_MI;
    Order = zeros(1, K);
    Order(1) = 1;
    A(max_idx) = []; % Remove selected feature

    if verbose
        fprintf('Selected features: #%d\n', S(1));
    end

    % Iteratively select features
    for t = 2:K
        CMI = zeros(1, length(A));
        ord = zeros(1, length(A));

        parfor i = 1:length(A)
            [redundancy, ord(i)] = n_order_total_redundancy_max(X(:, A(i)), X(:, S(1:t-1)), Y, n);
            CMI(i) = MI(A(i)) - redundancy;
        end

        [~, rank_idx] = max(CMI);
        S(t) = A(rank_idx);
        Order(t) = ord(rank_idx);
        CMIapp(t) = CMI(rank_idx);
        A(rank_idx) = []; % Remove selected feature

        if verbose
            fprintf('Selected features: %s\n', mat2str(S(1:t)));
        end
    end

    tElapsed = toc(tStart); % Compute elapsed time
    par.S = S;
    par.tElapsed = tElapsed;
    par.CMIapp = CMIapp;
end

function [redundancy, it_order] = n_order_total_redundancy_max(Xk, S, Y, order)
    % Compute n-order total redundancy maximum
    ns = size(S, 2);
    max_Rn = -Inf;
    ind_selected = [];

    % First order
    for i = 1:ns
        mi_ = mutual_info(Xk, S(:, i));
        cmi_ = cond_mutual_info(Xk, S(:, i), Y);
        if mi_ - cmi_ > max_Rn
            max_Rn = mi_ - cmi_;
            best_idx = i;
        end
    end
    ind_selected = [ind_selected, best_idx];
    redundancy = max_Rn;
    it_order = 1;

    % Higher orders
    for j = 2:min(ns, order)
        max_Rn = -Inf;
        for i = 1:ns
            if ~ismember(i, ind_selected)
                mi_ = mutual_info(Xk, [S(:, i), S(:, ind_selected)]);
                cmi_ = cond_mutual_info(Xk, [S(:, i), S(:, ind_selected)], Y);
                if mi_ - cmi_ > max_Rn
                    max_Rn = mi_ - cmi_;
                    best_idx = i;
                end
            end
        end
        ind_selected = [ind_selected, best_idx];
        redundancy = max_Rn;
        if redundancy / max_Rn < 0.05
            break;
        end
    end
end

function MI = mutual_info(X, Y)
    % Compute mutual information between X and Y
    if size(X, 2) > 1
        X = MIToolboxMex(3, X);
    end
    if size(Y, 2) > 1
        Y = MIToolboxMex(3, Y);
    end
    [p12, p1, p2] = estpab(X, Y);
    MI = estmutualinfo_indjs(p12, p1, p2, size(X, 1));
    if isnan(MI) || MI < 0
        MI = 0;
    end
end

function CMI = cond_mutual_info(X, Y, Z)
    % Compute conditional mutual information
    MI_XZ = mutual_info(X, [Y, Z]);
    MI_YZ = mutual_info(Y, Z);
    CMI = MI_XZ - MI_YZ;
    if isnan(CMI) || CMI < 0
        CMI = 0;
    end
end
