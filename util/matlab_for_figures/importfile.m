function markovmodelBLEURTscores = importfile(filename, dataLines)
%IMPORTFILE Import data from a text file
%  MARKOVMODELBLEURTSCORES = IMPORTFILE(FILENAME) reads data from text
%  file FILENAME for the default selection.  Returns the numeric data.
%
%  MARKOVMODELBLEURTSCORES = IMPORTFILE(FILE, DATALINES) reads data for
%  the specified row interval(s) of text file FILENAME. Specify
%  DATALINES as a positive scalar integer or a N-by-2 array of positive
%  scalar integers for dis-contiguous row intervals.
%
%  Example:
%  markovmodelBLEURTscores = importfile("C:\Users\Jasper\Dropbox\RUG - ME\DL - Deep Learning\Practical 2\results\MarkovModel\markov_model_BLEURT_scores.txt", [1, Inf]);
%
%  See also READTABLE.
%
% Auto-generated by MATLAB on 08-Apr-2021 18:21:04

%% Input handling

% If dataLines is not specified, define defaults
if nargin < 2
    dataLines = [1, Inf];
end

%% Setup the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 1);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = "VarName1";
opts.VariableTypes = "double";

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
markovmodelBLEURTscores = readtable(filename, opts);

%% Convert to output type
markovmodelBLEURTscores = table2array(markovmodelBLEURTscores);
end