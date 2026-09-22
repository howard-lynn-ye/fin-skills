function run_native(upstream, output_dir)
% Reproduction harness for the unmodified Luo/Huang model, commit recorded below.
% SPDX-License-Identifier: GPL-3.0-or-later
% Original model copyright 2024 Junjie Luo, Cheng Huang, Mark J. Schnitzer.
% Harness added 2026-09-21. No fitting or trading adaptation occurs here.
assert(~isfolder(output_dir), 'Refusing existing output directory');
mkdir(output_dir);
addpath(fullfile(upstream, 'matlab_code', 'model_functions'));
addpath(fullfile(upstream, 'matlab_code', 'output_functions'));
rng(20260921, 'twister');
runtime = version;
upstream_commit = '5d7c08a9a88f923169a0c3008aca68af421e9a7f';
t0 = tic;
for modules = [2 3]
    d = load(fullfile(upstream, 'data_and_parameters', ...
        sprintf('Dx_steady_state_nonlinear_3_27-Mar-2023_%dmodules.mat', modules)));
    [p, ~] = experimental_condition([]);
    p.session(3).n = 3;
    p.rest_t = [3600-250-300; 3600*2-250; 3600*21-250];
    p.session_list = p.session_list([1:end,end-1,end,end-1,end]);
    [~, protocol] = experimental_condition(p);
    for k = [4 16 20 32], protocol(k).t_length = 300; end
    n = size(d.para_rand, 2);
    responses = zeros(6, 2, 6, 2, n);
    central = zeros(6, 2, 6, 2);
    central_trace = cell(2,1);
    for sample = 0:n
        if sample == 0, v = d.para_mu; else, v = d.para_rand(:,sample); end
        params = parameter_vec2mat(v, d.mat_lu_cell);
        for odor_pair = 1:2
            curr = params;
            if odor_pair == 1, ix = 1:6; else, ix = [7:9 4:6]; end
            curr{1} = params{1}(1,ix);
            if sample == 0
                [central(:,:,:,odor_pair), central_trace{odor_pair}] = ...
                    Dx_steady_state_MBON_0301_2023(curr, protocol);
            else
                responses(:,:,:,odor_pair,sample) = ...
                    Dx_steady_state_MBON_0301_2023(curr, protocol);
            end
        end
        if mod(sample,1000)==0
            fprintf('native modules=%d sample=%d/%d elapsed=%.1fs\n', ...
                modules, sample, n, toc(t0));
        end
    end
    probabilities = [0.5, 0.5*erfc(1/sqrt(2)), 0.5*erfc(-1/sqrt(2))];
    bands = prctile(responses, probabilities*100, 5);
    save(fullfile(output_dir, sprintf('native_%dmodules.mat',modules)), ...
        'responses','central','central_trace','bands','probabilities','protocol', ...
        'runtime','upstream_commit','-v7');
end
% Fig. 5e's complete point-estimate grid: intact and feedback-disconnected.
d = load(fullfile(upstream,'data_and_parameters', ...
    'Dx_steady_state_nonlinear_3_27-Mar-2023_3modules.mat'));
training_counts = 3:3:15;
rest_seconds = (900:900:10800)-135;
feedback_grid = zeros(6,2,2,numel(rest_seconds),numel(training_counts),2);
for arm = 1:2
    v = d.para_mu;
    if arm == 2, v([12,13,15,18,19]) = 0; end
    params = reset_parameter_by_valance(v,d.mat_lu_cell,0);
    for j = 1:numel(training_counts)
        for k = 1:numel(rest_seconds)
            [p,~] = experimental_condition([]);
            p.session_list = {'imaging';'training';'rest';'imaging'};
            p.session(3).n = training_counts(j);
            p.rest_t = rest_seconds(k);
            [~, protocol_e] = experimental_condition(p);
            protocol_e(4).t_length = 300;
            feedback_grid(:,:,:,k,j,arm) = ...
                Dx_steady_state_MBON_0301_2023(params,protocol_e);
        end
    end
end
save(fullfile(output_dir,'native_feedback.mat'), 'feedback_grid', ...
    'training_counts','rest_seconds','runtime','upstream_commit','-v7');
elapsed_seconds = toc(t0);
save(fullfile(output_dir,'native_complete.mat'),'elapsed_seconds','runtime','upstream_commit','-v7');
fprintf('NATIVE_COMPLETE %.1fs\n',elapsed_seconds);
end
