%By Jasper van Vliet
%What: figures for DL practical 2
%when: april 2021

%% start
%clear all
dir = strcat(pwd,"\BLEURT\");

%BLEURT scores
B.markov =  importfile(strcat(dir,"markov_model_BLEURT_scores.txt"), [1, Inf]);
B.lSTMw =   importfile(strcat(dir,"lstm_with_emb_BLEURT_scores.txt"), [1, Inf]);
B.lSTMwo =  importfile(strcat(dir,"lstm_without_emb_BLEURT_scores.txt"), [1, Inf]);
B.gPT2 =    importfile(strcat(dir,"gpt2_BLEURT_scores.txt"), [1, Inf]);
%BLEURT txt
T(:,1) =  importtxt(strcat(dir,"markov_output_starting_with_HUGE.txt"), [1, Inf]);
T(:,2) =   importtxt(strcat(dir,"07_lstm_with_embedding_output_HUGE.txt"), [1, Inf]);
T(:,3) =  importtxt(strcat(dir,"07_lstm_output_HUGE.txt"), [1, Inf]);
T(:,4) =    importtxt(strcat(dir,"top_k_sampled_gpt2_HUGE.txt"), [1, Inf]);

b(:,1) = B.markov;
b(:,2) = B.lSTMw;
b(:,3) = B.lSTMwo;
b(:,4) = B.gPT2;



%% BLEURT results
average = mean(b)
standdev= std(b,1) %normalized by N (not N-1)

%raw data
figure
plot(B.markov);hold on
plot(B.lSTMw);grid on
plot(B.lSTMwo);
plot(B.gPT2);

%histogram
for i = 1:size(b,2)
figure
hist(b(:,i));
end

%% sort HUGE 'candidates' based on BLEURT scores
[~,I] = sort(b);
for i = 1:4
    Tsort(:,i) = T(I(:,i),i);
end

%BLEURT Results, best to worse
for i = 1:4
    T_best_to_worst(:,i) = flip(Tsort(:,i));
end
t = '_HUGE_BLEURT_sorted_best_to_worst.txt';
filenames ={strcat('markov',t);
            strcat('LSTM_with_embedding',t);
            strcat('LSTM_without_embedding',t);
            strcat('GPT2',t)};
for i = 1:4
    filePh = fopen(string(filenames(i)),'w');
    fprintf(filePh,'%s\n',string(T_best_to_worst(:,i)));
end
fclose('all');

%% make loss fig
path = strcat(pwd,"\csv files\");
dat = table2array(importcsv(strcat(path,"gpt2_finetuning_training_loss.csv")));
load('LSTM_loss_data.mat')

%gpt training loss
figure
plot(dat(1:15,1),dat(1:15,2))
xlabel('Steps')
ylabel('Training Loss')
title('Training Loss GPT-2')
ylim([0 2.25])

%lstm training loss\
figure
plot(lstmw)
xlabel('Epochs')
ylabel('Training Loss')
title('Training Loss LSTM with embedding')
hold on
plot(lstmwo)
xlabel('Epochs')
ylabel('Training Loss')
title('Training Loss LSTM without embedding')
legend('with','without')





    