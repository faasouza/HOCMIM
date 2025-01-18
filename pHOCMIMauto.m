function [S, tElapsed, CMIapp, par, Order]  = pHOCMIMauto(X, Y, K, n, fast, verbose)
% paralell implementation of HOCMIM
% Code for paper pr2022_FranciscoSouza.tex
% Author: Francisco Alexandre
% 
% version 1.0
% date: 11/2020

% This is the implementation code of [1].
% Arguments:
% X: Train data in colune wise (M X N), where:
%    m: samples
%    n: input size
% Y: size (n X 1)
% K: number of features to select
% n: order of HOCMIM approximation
% fast: true or false
%   . true: implements the fast approximation of solution by the
%   Markov Model assumption.
%   . false: implements the evaluation of all features
% verbose:
%   . true: print verbose
%   . false: hide verbose

if nargin < 6
    verbose = false;
end

% The code is self contained, it only uses this single file.

N = size(X,2);

if K>N
    K = N;
end

% compute model time
tStart=tic;
% indices of variables
A = 1:N; 
% 
% S = zeros(1,K);
%
CMIapp = zeros(1,K);

% Select first variable based on MI
MI = zeros(1,N);
pho = zeros(1,N);
for i=1:N
    MI(i) = mutual_info(X(:,i),Y);
    pho(i) = MI(i);
end
Order(1)=1;

% first term for CMI 
Rn(1) = max(MI);
CMIapp(1) = max(MI);

% rank mutual information
[~,max_MI] = max(MI);

if verbose
    disp('Selected features: ')
    disp(['#' num2str(max_MI)])
end
ord(i) =1;
S(1) = A(max_MI);
A(max_MI) = [];
%pho(max_MI) = [];

for t = 2:K
    pho_max = -1;
    CMI=0;
    ord=0;
    parfor i=1:length(A)
        [tmp,ord(i)] = n_order_total_redundancy_max(X(:,A(i)),X(:,S),Y,n);
        CMI(i) = pho(A(i)) - tmp ;
        %  mutual_info(X(:,A(i)),Y)
    end
    [~,rankX] = max(CMI);
    CMIapp(t) = max(CMI);
    
    % add to the set of selected variables
    S(t) = A(rankX);
    Order(t) = ord(rankX);
    A(rankX) = [];
%    pho(rankX) = [];
    
    if verbose
        S_to_print = '';
        for i=1:length(S)
           S_to_print = [S_to_print '#' num2str(S(i)) '  '];
        end
        disp(S_to_print)
    end
end

tElapsed=toc(tStart);

par.S = S;
par.tElapsed = tElapsed;
par.CMIapp = CMIapp;

end

function [Rn, it_order] = n_order_total_redundancy_max(Xk,S,Y,order)

ns = size(S,2);
ny = size(Y,2);
max_Rn = -1e6;
ind_selected_features=[];
L=order;
j=1;

Rn_max = mutual_info(Xk,Y);
EntY = mutual_info(Y,Y);

% first order
for i=1:ns
    mi_ = mutual_info(Xk,S(:,i));
    cmi_ = cond_mutual_info(Xk,S(:,i),Y);
    if mi_-cmi_ > max_Rn
        max_Rn = mi_-cmi_;
        ind_best_feature = i;
    end
end
ind_selected_features = [ind_selected_features ind_best_feature];
Rn(1) = max_Rn;
it_order = 1;


% if abs(max_Rn/Rn_max)<1e-2 
%     %Rn = min([Rn_max,sum(Rn)]);
%     it_order = 1;
%     return;
% end

% rest of order
% if ns >1
for j=2:min([ns L])
    max_Rn =-1e12;
    ind_best_feature=[];
    
    %tmpS = MIToolboxMex(3,S(:,ind_selected_features));
    %tmpSY = MIToolboxMex(3,[Y S(:,ind_selected_features)]);
    
    for i=1:ns
        if ~(sum(i==ind_selected_features)>0)
            mi_= mutual_info(Xk, S(:,[i ind_selected_features]));
            cmi_= cond_mutual_info(Xk, S(:,[i ind_selected_features]), Y);
            if mi_-cmi_ > max_Rn
                max_Rn = mi_-cmi_;
                ind_best_feature = i;
                
            end
        end
    end
    ind_selected_features = [ind_selected_features ind_best_feature];

    if isempty(ind_best_feature)
        Rn(j) =0;
    else
        Rn(j) = max_Rn;
    end
    
    if ((Rn(end))>=((1-0.05)*Rn_max)) || (Rn(end)>=0 && Rn(end)<0.05*Rn_max)
        break
    end
    
%     if (Rn_max/EntY) > 5e-2
%         if (sum(Rn)/Rn_max) > 1-5e-2
%             break
%          end
%         % if sum(Rn) > 0-1e-2 && j>5
%         %    break
%         % end
% %         top=0;
% %     else
% %         if (sum(Rn)/Rn_max) > 1-5e-2
% %             break
% %          end
%     end
    
    
    
%     if (sum(Rn)/Rn_max)> 1 && abs(Rn(end)/Rn_max)<1e-1
%         break
%     end
%     if Rn_max <1e-3
%         if (j>3 && Rn(end)>0) || (j>3 && sum(Rn)>0)
%             break;
%         end
%     else

end


%Rn_F = Rn;


%Rn_B = flip(n_order_total_redundancy_min(Xk,S(:,ind_selected_features),Y));
if isempty(j), j=1; end
it_order = j;
Rn = Rn(end);%sum(Rn);%sum(Rn_F(1:min([ns order])));

%disp(mutual_info(S(:,ind_selected_features), Xk)-cond_mutual_info(S(:,ind_selected_features), Xk, [Y ]))
%disp(Rn)
end

function [Rn] = n_order_total_redundancy_min(Xk,S,Y)
% backward selection. from the selected set in the forward grid, perform
% the opposite direction selection

ns = size(S,2);
ny = size(Y,2);
min_Rn = 1e6;
ind_selected_features=[];
A = 1:ns;

% first order
for j=1:ns
    min_Rn =1e6;
    ind_best_feature=[];
    for i=1:length(A)
        tmp = A;
        tmp(i)=[];
        if isempty(tmp)
            mi_= mutual_info(S(:,A(i)), Xk);
            cmi_= cond_mutual_info(S(:,A(i)), Xk, [Y]);
        else
        mi_= cond_mutual_info(S(:,A(i)), Xk, S(:,tmp));
        cmi_= cond_mutual_info(S(:,A(i)), Xk, [Y S(:,tmp)]);
        end
        if mi_-cmi_ < min_Rn
            min_Rn = mi_-cmi_;
            ind_best_feature = i;
        end
    end
    ind_selected_features = [ind_selected_features A(ind_best_feature)];
    Rn(j) = min_Rn;
    A(ind_best_feature)=[];
end

% rest of order
% if ns >1

end

function [Rn] = n_order_total_redundancy_sep_max(Xk,S,Y,order)

ns = size(S,2);
ny = size(Y,2);
max_mi = -1e6;
min_cmi = 1e6;
ind_selected_features_mi=[];
ind_selected_features_cmi=[];
order =10;
Rn_max = mutual_info(Xk,Y);
    
    
% first order
for i=1:ns
    mi_ = mutual_info( Xk, S(:,i));
    cmi_ = cond_mutual_info(Xk, S(:,i), Y);
    if mi_ > max_mi
        max_mi = mi_;
        ind_best_feature_mi = i;
    end
    if cmi_ < min_cmi
        min_cmi = cmi_;
        ind_best_feature_cmi = i;
    end
end
ind_selected_features_mi = [ind_selected_features_mi ind_best_feature_mi];
ind_selected_features_cmi = [ind_selected_features_cmi ind_best_feature_cmi];
Rn(1) = max_mi-min_cmi;

% rest of order
% if ns >1
for j=2:min([ns order])
    max_mi = -1e6;
    min_cmi = 1e6;
    ind_best_feature_mi=[];
    ind_best_feature_cmi=[];
    for i=1:ns
        if ~(sum(i==ind_selected_features_mi)>0)
            mi_= cond_mutual_info(Xk, S(:,i), S(:,ind_selected_features_mi));
            if mi_ > max_mi
                max_mi = mi_;
                ind_best_feature_mi = i;
            end
        end
        if ~(sum(i==ind_selected_features_cmi)>0)
            cmi_= cond_mutual_info(Xk, S(:,i), [Y S(:,ind_selected_features_cmi)]);
            if cmi_ < min_cmi
                min_cmi = cmi_;
                ind_best_feature_cmi = i;
            end
        end
    end
    ind_selected_features_mi = [ind_selected_features_mi ind_best_feature_mi];
    ind_selected_features_cmi = [ind_selected_features_cmi ind_best_feature_cmi];

    if isempty(ind_best_feature_mi) || isempty(ind_best_feature_cmi)
        Rn(j) =0;
    else
        Rn(j) = max_mi - min_cmi;
    end

    if sum(Rn/Rn_max)>0.999
       % Rn = min([Rn_max,sum(Rn)]);
        break;
    end
    
end

Rn=sum(Rn);

end


function MI = mutual_info(X, Y)
% Describe ....
% Adapted from authors implementation.

% Reference: 
%   [2]: K. Sechidis, et al. Sechidis, K. et al. "Efficient feature selection 
%        using shrinkage estimators". Machine Learning Journal (2019).
%        doi: 10.1007/s10994-019-05795-1.
% github: 
%   . https://github.com/sechidis/2019-MLJ-Efficient-feature-selection-using-shrinkage-estimators

% MI = mi(X,Y);
% 
% MI = mi(X,Y);
% return;

if (size(X,2)>1)
    X= MIToolboxMex(3,X);
end
if (size(Y,2)>1)
    Y= MIToolboxMex(3,Y);
end


[p12, p1, p2] = estpab(X,Y);

MI = estmutualinfo_indjs(p12,p1,p2,size(X,1));

if isnan(MI) || MI<0
    MI=0;
end

end

function CMI= cond_mutual_info( X, Y, Z)
% Describe ....
% Adapted from authors implementation.

% Reference: 
%   [2]: K. Sechidis, et al. Sechidis, K. et al. "Efficient feature selection 
%        using shrinkage estimators". Machine Learning Journal (2019).
%        doi: 10.1007/s10994-019-05795-1.
% github: 
%   . https://github.com/sechidis/2019-MLJ-Efficient-feature-selection-using-shrinkage-estimators

% CMI = cmi(X,Y,Z);
% 
% return
%XZ= MIToolboxMex(3,[X Z]);
%Y= MIToolboxMex(3,[Y]);
%Z= MIToolboxMex(3,[Z]);

% CMI = cmi(X,Y,Z);
% return;


MI_Y_XZ = mutual_info(Y,[X Z]);
MI_X_YZ = mutual_info(X,[Y Z]);
% [p12, p1, p2] = estpab(XZ,Y);
% MI_XZ_Y = estmutualinfo_indjs(p12,p1,p2);

MI_Y_Z = mutual_info([Z],Y);
MI_X_Z = mutual_info(X,Z);

% [p12, p1, p2] = estpab(Z,Y);
% MI_Z_Y = estmutualinfo_indjs(p12,p1,p2);

CMI = MI_Y_XZ - MI_Y_Z;

CMI2 = MI_X_YZ - MI_X_Z;

CMI=max([CMI, CMI2]);
if isnan(CMI) || CMI<0
    CMI=0;
end


end

