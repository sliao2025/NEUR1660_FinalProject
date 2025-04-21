clearvars 

%% Task Parameters
runs = 5; %how many times it will go through each trial block

num_blocks = 3;
nStates = num_blocks;
low_rewards = [5 10 20]; %possible rewards in the low reward block
high_rewards = [20 40 80]; %possible rewards in the high reward block
mixed_rewards = [5 10 20 40 80]; %possible rewards in the mixed block

% turn possibilities into probabilities
highDist=[0, 0, 1, 1, 1];
lowDist=[ 1, 1, 1, 0, 0];
midDist=[ 1, 1, 1, 1, 1];

highDist=highDist./sum(highDist);
lowDist=lowDist./sum(lowDist);
midDist=midDist./sum(midDist);
allProb = [lowDist; highDist; midDist];
flatDist=ones(nStates,1)./nStates;

num_trials = 40;
haz=1/40;

all_rewards = nan(num_blocks, num_trials, runs); %record of all of the rewards that the animal received (columns are trials, rows are blocks, frames are runs)
initiation_times = nan(num_blocks, num_trials, runs); % initiation times

trial_initiation_times = nan(1,num_blocks*num_trials*runs);
trial_rewards = nan(1,num_blocks*num_trials*runs);
state_order = [];
RPE_first_ten = nan(num_blocks,10, runs); %reward prediction errors for the first 10 trials of every block
beliefs_first_ten = nan(num_blocks,11, runs); %record the beliefs for the firs 10 trials of every block and the one trial before them (bc we're gonna calc the change later)

opt_out = nan; %this will change once we have the model

%% Model Parameters

state_neurons = rand(3,1)/10; %initialize the initial state values between 0-0.1
synaptic_lr = 0.3;
state_lr = 0.1;
D = 0.1; %Scale factor for initiation times

%% Runs
weight_matrix = rand(3,1);
epsilon = 1e-3; % to avoid division by zero

pState = flatDist;
trial_counter = 1;
for run = 1:runs
    block_order = randperm(3); %the order that blocks will be presented in for this run 
    
    for b = 1:length(block_order)
        possible_rewards = [];
        if block_order(b) == 1 %if the bth block is 1, the possible rewards are the low value ones
            possible_rewards = low_rewards;
          
        elseif block_order(b) == 2 %if the bth block is 2, the possible rewards are the high value ones
            possible_rewards = high_rewards;
            
        elseif block_order(b) == 3 %if the bth block is 3, the possible rewards are the mixed value ones
            possible_rewards = mixed_rewards;
            
        end

        for t = 1:num_trials
            ground_truth_states(trial_counter) = block_order(b);
            
            reward_index = randi(length(possible_rewards), 1);
            reward_val   = possible_rewards(reward_index);    
            global_index = find([5 10 20 40 80] == reward_val);
            
            allPState(trial_counter,:) = pState;
            
            %state inference
            likeRew = allProb(:, global_index);
            prior = pState.*(1-haz)+flatDist.*haz;
            pState=likeRew.*prior;
            pState=pState./sum(pState);

            %updating weights and state values
            trial_reward_offer = possible_rewards(reward_index)/80; %the reward that could be represented on this trial
            output_act = (weight_matrix)'*state_neurons; %should be 1x1
            output_act = max(output_act,0);
            RPE = trial_reward_offer - output_act;
            weight_matrix = weight_matrix.*(1-synaptic_lr) + (synaptic_lr * RPE * pState); %existing weights + weight update
            state_neurons = state_neurons.*pState*(1-state_lr) + state_lr*RPE; %existing state + state update

            % Compute and store initiation time (higher activation = faster initiation)
            initiation_time = D / (output_act + epsilon);
            initiation_times_raw(trial_counter) = initiation_time;
            initiation_times(block_order(b), t, run) = initiation_time;
            trial_initiation_times(trial_counter) = initiation_time;
            trial_rewards(trial_counter) = trial_reward_offer;

            if t == 40
                state_order(end+1) = block_order(b);
            end
            
            if t<11
                RPE_first_ten(b, t, run) = RPE;
                if run == 1 && b ==1 && t ==1
                    beliefs_first_ten(b,t,run) = 0;
                else
                    [row,column] = max(allPState(trial_counter-1,:));
                    beliefs_first_ten(b,t,run) = column;
                    [row,column] = max(allPState(trial_counter,:));
                    beliefs_first_ten(b,t+1,run) = column;
                end
            end

            all_rewards(block_order(b), t, run) = trial_reward_offer; %this is set up so that the row (block index) is constant despite the fact that the order of the blocks is changing (so the first row will always be low reward, second high, and third mixed)
            trial_counter = trial_counter + 1;

        end
    end
end

%% Plot Initiation Times
figure;
hold on;
colors = lines(num_blocks);
labels = {'Low Reward Block', 'High Reward Block', 'Mixed Reward Block'};

for block = 1:num_blocks
    avg_initiation = squeeze(mean(initiation_times(block, :, :), 3)); % avg across runs
    plot(avg_initiation, 'LineWidth', 2, 'Color', colors(block,:));
end

xlabel('Trial');
ylabel('Initiation Time');
legend(labels);
title('Average Initiation Time Across Trials Per Block');
grid on;

%% Calculate the Mean Initiation Time Per Block Type
% For each block, average initiation time across all trials and runs
mean_initiation_time = squeeze(mean(mean(initiation_times, 2), 3));

%% Plot the Mean Initiation Times Per Block as Dots with Custom Colors
figure;
hold on;
plot(1, mean_initiation_time(1), 'o', 'Color', 'b', 'MarkerFaceColor', 'b', 'MarkerSize', 10);
plot(2, mean_initiation_time(3), 'o', 'Color', [0.5 0.5 0.5], 'MarkerFaceColor', [0.5 0.5 0.5], 'MarkerSize', 10);
plot(3, mean_initiation_time(2), 'o', 'Color', 'r', 'MarkerFaceColor', 'r', 'MarkerSize', 10);


% Set x-axis tick labels corresponding to the block types
set(gca, 'XTick', 1:3, 'XTickLabel', {'Low Reward Block', 'Mixed Reward Block', 'High Reward Block'});
xlabel('Block Type');
ylabel('Mean Initiation Time');
title('Mean Initiation Time per Block Type');
grid on;

%% Plotting States
% --- Flatten pState across runs/trials ---
flatP = reshape(allPState, [], 3);   % size = (runs*num_trials)×3 = 600×3

% --- Plot as three rows of vertical lines ---
figure; hold on;
numTrialsTotal = size(flatP,1);

for s = 1:3
    for t = 1:numTrialsTotal
        intensity = flatP(t,s);          % pState for state s at trial t
        grayLevel  = 1 - intensity;      % 0 = black (high prob), 1 = white (low prob)
        line([t t], [s-0.4, s+0.4], ...   % small vertical segment
             'Color', [grayLevel grayLevel grayLevel], ...
             'LineWidth', 2);
    end
end

% --- Tidy up axes ---
xlim([1 numTrialsTotal]);
ylim([0.5 3.5]);
yticks(1:3);
yticklabels({'Low','High','Mixed'});   % or whatever your state‐order is
xlabel('Trial');
ylabel('State');
title('Belief (pState) Over Trials for Each State');

yyaxis right
plot(1:numTrialsTotal, initiation_times_raw, 'b-', 'LineWidth', 1.5);
ylabel('Initiation Time');
grid on;

% 1) Flatten all_rewards into 1×(runs*num_trials)
flatR = reshape(ground_truth_states, 1, []);    % size = 1×200

% 2) Plot exactly as before—but now for the actual state
N = length(ground_truth_states);
figure; hold on;
for s = 1:3
    idx = find(ground_truth_states == s);
    for t = idx'
        line([t t], [s-0.4, s+0.4], 'Color', [0 0 0], 'LineWidth', 2);
    end
end

xlim([1 N]);
ylim([0.5 3.5]);
yticks(1:3);
yticklabels({'Low','High','Mixed'});
xlabel('Trial');
ylabel('State');
title('Actual Block Sequence');


%% Plot the trial initiation time as a function of trial from block switch (figure 2A)

%get the variables
switch_trials = num_trials:num_trials:size(allPState,1);
switch_trials = switch_trials+1;
switch_trials = switch_trials(1:end-1);
lowtomixed = [];
hightomixed = [];
for s = 1:length(switch_trials)
    thirtybefore = switch_trials(s) - 30; 
    fourtyafter = switch_trials(s) + (40-1);
    if state_order(s) == 1 &&  state_order(s+1)== 3
        lowtomixed(end+1,:) = trial_initiation_times(thirtybefore:fourtyafter);
    elseif state_order(s) == 2 &&  state_order(s+1)== 3
        hightomixed(end+1,:) = trial_initiation_times(thirtybefore:fourtyafter);
    end
end
mean_lowtomixed = mean(lowtomixed,1);
mean_hightomixed = mean(hightomixed,1);

%plot
figure;
hold on;
plot(-30:39, mean_lowtomixed, 'Color', 'b', 'LineWidth', 2);
plot(-30:39, mean_hightomixed, 'Color', 'r', 'LineWidth', 2);
xlabel('Trial fromm Block Switch');
ylabel('Trial Initiation Time');
hold off;


%plot wih "causal filter" (what they did in the paper)
windowSize = 10;
filteredmeanlowtomixed= nan(size(mean_lowtomixed));
filteredmeanhightomixed = nan(size(mean_hightomixed));
for n = windowSize:length(mean_lowtomixed)
    filteredmeanlowtomixed(n) = mean(mean_lowtomixed(n-windowSize+1:n));
    filteredmeanhightomixed(n) = mean(mean_hightomixed(n-windowSize+1:n));
end
figure;
hold on;
plot(-30:39, filteredmeanlowtomixed, 'Color', 'b', 'LineWidth', 2);
plot(-30:39, filteredmeanhightomixed, 'Color', 'r', 'LineWidth', 2);
xlabel('Trial from Block Switch');
xlim([-20 40]);
ylabel('Trial Initiation Time');
title('with 10 trial filter')
hold off;

%% Initiation times as a function of RPE sign


belief_change_first_ten = diff(beliefs_first_ten,1,2); 
median_belief_change = median(belief_change_first_ten, "all");

low_belief_change = [];
high_belief_change = [];

for run = 1:runs
    for b = 1:num_blocks
        for t = 1:10
            flat_index = (run - 1) * num_blocks * num_trials + (b - 1) * num_trials + t;

            if flat_index <= 1 || isnan(RPE_first_ten(b,t,run))
                continue; % skip invalid or NaN RPE
            end

            delta_init_time = trial_initiation_times(flat_index) - trial_initiation_times(flat_index - 1);

            if belief_change_first_ten(b,t,run) < median_belief_change
                if RPE_first_ten(b,t,run) < 0
                    low_belief_change(end+1,1) = delta_init_time;
                elseif RPE_first_ten(b,t,run) > 0
                    low_belief_change(end+1,2) = delta_init_time;
                end
            else
                if RPE_first_ten(b,t,run) < 0
                    high_belief_change(end+1,1) = delta_init_time;
                elseif RPE_first_ten(b,t,run) > 0
                    high_belief_change(end+1,2) = delta_init_time;
                end
            end
        end
    end
end

mean_lowbeliefchange = mean(low_belief_change, 1, 'omitnan');
mean_highbeliefchange = mean(high_belief_change, 1, 'omitnan');


%now plot
figure;
hold on;
plot([1 2], mean_lowbeliefchange.*10, 'Color', 'b', 'LineWidth', 2, 'DisplayName', 'Low Change in Belief');
plot([1 2], mean_highbeliefchange.*10, 'Color', 'r', 'LineWidth', 2, 'DisplayName', 'High Change in Belief');
legend('show');
ylabel('Change in Initiation Time (x10^-1)');
xlim([0 3]);
xticks([1 2]);
xticklabels({'RPE<0', 'RPE>0'});
hold off;

