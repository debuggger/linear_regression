function [] = proj2()

    
    real_data = load('data.mat', '-mat');
    rdata = real_data.data;
    rlabel = double(real_data.label);
    %{
    [rdata, rlabel] = importdata(real_data_file);
    %}
    
    syn_data = load('synthetic.mat', '-mat');
    sdata = syn_data.x';
    slabel = syn_data.t;
    
    a=0;
    output = 'proj2.mat';
    save(output, 'a');
    %train_real(rdata, rlabel, output);
    train_synthetic(sdata, slabel, output);

end


function [err] = compute_error(w, design_mat, indices, label, samples)
    sampleIndices = randi([indices(1) indices(length(indices))], 1, samples);
    err = sqrt(sum((w*design_mat(indices,:)' - label(indices)').^2)/length(indices));
end

function [w ,dw, erms_train, erms_validation ,erms_test, etaList] = stochastic_local(design, t, train, validation, test, w01, eta, lambda, step_incr, wml, numIterations)
    w = [w01'];
    dw = [];
    
    etaList = [];
    
    eta = eta;
    min_err_list = [];
    min_iter_list = [];
    min_wt_list = [];
    
    wtaunext = w01';
    train_iteration = 0;
    err_list = [];
    
    min_err = 1;
    min_iter = 1;
    min_wt = wtaunext;
    
    for iter=1:numIterations
        %fprintf('iter:%d\n', iter);
        for i = train
            train_iteration = train_iteration+1;
            phix = design(i, :);

            ED = -1*(t(i) - wtaunext*phix')*phix;
            
            deltaEtau = -1*eta*(ED + lambda*wtaunext);
            
            wtaunext = wtaunext + deltaEtau;
            
            dw = [dw deltaEtau'];
            
            etaList = [etaList, eta];
            
            %if(mod(train_iteration, 500) == 0)
                err_itr = compute_error(wtaunext, design, train, t, length(train));
                err_list = [err_list err_itr];
                    
                if(err_itr < min_err)
                    %err_list = [err_list err_itr];
                    min_err = err_itr;
                    min_wt = wtaunext;
                    min_iter = train_iteration;
                end
            %end
            
            
        end
        min_err_list = [min_err_list; min_err];
        min_wt_list = [min_wt_list; min_wt];
        min_iter_list = [min_iter_list; min_iter];
    end
    save('err_list.mat', 'min_wt_list', 'min_iter_list', 'min_err_list', 'err_list');
    %figure; plot(err_list);
     
    [min_val, min_index] = min(min_err_list);
    
    wtau_final = min_wt_list(min_index, :);
    norm(wtau_final- wml')
    %sqrt(sum((wtau_final-wml').^2))
    
    min_iter_list(min_index);
    dw = dw(:, 1:min_iter_list(min_index));
    etaList = etaList(:, 1:min_iter_list(min_index));
    
    result_train = wtau_final*design(train, :)';
    result_validation = wtau_final*design(validation, :)';
    result_test = wtau_final*design(test, :)';
    
    sose_train = 0.5*sum((t(train) - result_train').^2);
    sose_validation = 0.5*sum((t(validation) - result_validation').^2);
    sose_test = 0.5*sum((t(test) - result_test').^2);
    
    
    erms_train = sqrt(2*sose_train/length(train));
    erms_validation = sqrt(2*sose_validation/length(validation));
    erms_test = sqrt(2*sose_test/length(test));
    
end


function [wml, design, erms_train, erms_validation, erms_test] = train_closed_form(data, label, train, validation, test, lambda, M, mean, Sigma)
    dim = length(data(1,:));
    n = length(data(:,1));
    
    fprintf('computing design matrix...');
    design = compute_design_matrix(data, mean, Sigma);
    fprintf('done\n');
    
    wml = inv((lambda*eye(M+1)) + design(train, :)'*design(train, :))*design(train, :)'*label(train);


    result_train = wml'*design(train, :)';
    result_validation = wml'*design(validation,:)';
    result_test = wml'*design(test,:)';

    err_train = 0.5*sum((result_train' - label(train)).^2); %+ 0.5*lambda*(wml'*wml);
    erms_train = sqrt(2*err_train/(length(train)));

    err_validation = 0.5*sum((result_validation' - label(validation)).^2); %+ 0.5*lambda*(wml'*wml);
    erms_validation = sqrt(2*err_validation/(length(validation)));
    
    err_test = 0.5*sum((result_test' - label(test)).^2); %+ 0.5*lambda*(wml'*wml);
    erms_test = sqrt(2*err_test/(length(test)));
end

function [] = train_real(data, label, output)
    
    dim = length(data(1,:));
    n = length(data(:,1));
    covariance = cov(data);
    covariance = eye(dim) + covariance;

    lambda = 0.0001;
    train = 1:int32(0.8*n);
    validation = int32((0.8*n)+1):int32(0.9*n);
    test = int32((0.9*n)+1):n;
    
        
    for i=5
       rng default;
       fprintf('kmeans start....');
       [id, mean] = kmeans(data, i);
       fprintf('done\n');
       clear('Sigma1');
       
       %[id, mean, Sigma1] = ccmv(data(train, :), i);
       Sigma1(:,:,1) = eye(dim);
       mean = [zeros(1,dim); mean];
       eta = 0.3;
            
       M1 = i+1;

       for j=1
           
            for k = 2:i+1
                Sigma1(:,:,k) = covariance .* j;
            end
           
            [wml, design, erms_train, erms_validation, erms_test] = train_closed_form(data, label, train, validation, test, lambda, i, mean, Sigma1);

            disp('===============Real=================')
            fprintf('Closed-form Model=%d, Covariance=%d, lambda=%f, eta=%f, Erms_train=%f, Erms_validation=%f Erms_test=%f \n', i, j, lambda, eta, erms_train, erms_validation, erms_test)

            mu1 = mean';
            lambda1 = lambda;
            trainInd1 = double(train');
            validInd1 = double(validation');
            w1 = wml;
            trainPer1 = erms_train;
            validPer1 = erms_validation;
            w01 = 0.0*ones(1,M1)';
            
            for eta=0.002
                [w_stochastic, dw1, erms_train_st, erms_validation_st, erms_test_st, eta1] = stochastic_local(design, label, train, validation, test, w01, eta, lambda1, 1, w1, 2);
                fprintf('Stochastic Model=%d,   Covariance=%d, lambda=%f, eta=%f, Erms_train=%f, Erms_validation=%f Erms_test=%f \n', i, j, lambda, eta, erms_train_st, erms_validation_st, erms_test_st)
                disp('====================================')    

            end
            
            save(strcat(output), 'w01', 'dw1', 'M1', 'mu1', 'Sigma1', 'lambda1', 'trainPer1', 'validPer1', 'validInd1', 'trainInd1', 'w1', 'eta1', '-append');
       end
    end
end

function [] = train_synthetic(data, label, output)
    dim = length(data(1,:));
    n = length(data(:,1));
    covariance = cov(data);
    covariance = eye(dim) + covariance;
    
    train = 1:int32(0.8*n);
    validation = int32((0.8*n)+1):int32(0.9*n);
    test = int32((0.9*n)+1):n;
    
    lambda = 1e-5;
    result_syn = [];
       
    for i=4
       fprintf('kmeans start....');
       rng default;
       [id, mean] = kmeans(data(train,:), i);
       fprintf('done\n');
       clear('Sigma2');
       Sigma2(:,:,1) = eye(dim);
       %[id, mean, Sigma2] = ccmv(data(train, :), i);
       mean = [zeros(1,dim); mean];
       
       M2 = i+1;
      
       for j=1
           
            for k = 2:i+1
                Sigma2(:,:,k) = covariance .* j;
            end
            
            for lambda=1e-4
                for eta=0.01
                    step=1;

                    [wml, design, erms_train, erms_validation, erms_test] = train_closed_form(data, label, train, validation, test, lambda, i, mean, Sigma2);
                    result_syn = [result_syn [i, lambda, erms_train, erms_validation]'];
                    
                    disp('===============Syn=================')
                    fprintf('Closed-form Model=%d, Covariance=%d, lambda=%f, eta=%f, Erms_train=%f, Erms_validation=%f Erms_test=%f \n', i, j, lambda, eta, erms_train, erms_validation, erms_test)
            
                    mu2 = mean';
                    lambda2 = lambda;
                    trainInd2 = double(train');
                    validInd2 = double(validation');
                    w2 = wml;
                    trainPer2 = erms_train;
                    validPer2 = erms_validation;
                    w02 = 0.0*ones(1,M2)';

                    [w_stochastic, dw2 , erms_train_st, erms_validation_st , erms_test_st, eta2] = stochastic_local(design, label, train, validation, test, w02, eta, lambda, step, w2, 80);
                    fprintf('Stochastic Model=%d,   Covariance=%d, lambda=%f, eta=%f, Erms_train=%f, Erms_validation=%f Erms_test=%f \n', i, j, lambda, eta, erms_train_st, erms_validation_st, erms_test_st)
                    
                    disp('====================================')
                    
                    save(strcat(output), 'result_syn', 'w02', 'dw2', 'M2', 'mu2', 'Sigma2', 'lambda2', 'trainPer2', 'validPer2', 'validInd2', 'trainInd2', 'w2', 'eta2', '-append');
                end
            end
       end
    end
    
    save('result_syn.mat', 'result_syn');
end

function phi = compute_design_matrix(X, mean, Sigma)
    phi = ones(length(X(:,1)),length(mean(:,1)));
    offset = 1e-3;
    
	for j = 2:length(mean(:,1))
        for i = 1:length(X(:,1))
            covariance = Sigma(:,:,j);
            if det(covariance) == 0
                covariance = offset*eye(length(covariance(1,:))) + covariance;
            end
			phi(i, j) = exp(-0.5*(X(i,:)-mean(j,:))*inv(covariance) * (X(i,:)-mean(j,:))');
		end
    end
end

function [data, label] = importdata(filename)
    data = [];
    label = [];
    fid = fopen(filename);
    tline = fgetl(fid);
    count = 0
    while ischar(tline)
        lineData = textscan(tline, '%d %s %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f %d:%f');
        data = [data; cell2mat(lineData(1,[4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94]))];
        label = [label; cell2mat(lineData(1,1))];
        count = count+1;
        lineData;
        tline = fgetl(fid);
    end
    
    data = double(data);
    label = double(label);
    
end
