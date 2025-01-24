(function() {
    if (typeof ace !== 'undefined' && ace.require) {
        // Define custom highlighter
        ace.define("ace/mode/custom_highlight_rules", ["require", "exports", "module", "ace/lib/oop", "ace/mode/text_highlight_rules"], function(require, exports, module) {
            var oop = require("ace/lib/oop");
            var TextHighlightRules = require("ace/mode/text_highlight_rules").TextHighlightRules;
            
            var CustomHighlightRules = function() {
                this.$rules = {
                    "start": [
                        {
                            token: "keyword",
                            regex: "\\b(model|task|mlModel|type|metric|start|load|from|predict|classification|regression|parameter|name|value|select|where|using|show|rule|if|then|and|or|with|goal)\\b"
                        },
                        {
                            token: "support.function",
                            regex: "\\b(RandomForest|DecisionTree|SVM|AutoML)\\b"
                        },
                        {
                            token: "support.constant",
                            regex: "\\b(rmse|mae|mse|accuracy|precision|recall|f1|auc|logloss|error|hamming_loss|jaccard|matthews_corrcoef|zero_one_loss|average_precision|neg_log_loss|neg_mean_absolute_error|neg_mean_squared_error|neg_mean_squared_log_error|neg_median_absolute_error|balanced_accuracy|adjusted_mutual_info_score|adjusted_rand_score|completeness_score|explained_variance|fowlkes_mallows_score|homogeneity_score|mutual_info_score|neg_brier_score|normalized_mutual_info_score|r2_score|rand_score|v_measure_score|all)\\b"
                        },
                        {
                            token: "support.type",
                            regex: "\\b(models|features|metrics)\\b"
                        },
                        {
                            token: "constant.numeric",
                            regex: "\\b\\d+(\\.\\d+)?\\b"
                        },
                        {
                            token: "string",
                            regex: '"(?:\\\\.|[^"\\\\])*"'
                        },
                        {
                            token: "comment",
                            regex: "\\/\\/.*$"
                        },
                        {
                            token: "punctuation.operator",
                            regex: "\\.|\\,|\\:|\\{|\\}|\\[|\\]|\\(|\\)|<|>|<=|>=|==|!="
                        },
                        {
                            token: "variable",
                            regex: "\\b[a-zA-Z_][a-zA-Z0-9_]*\\b"
                        }
                    ]
                };
            };
            oop.inherits(CustomHighlightRules, TextHighlightRules);
            exports.CustomHighlightRules = CustomHighlightRules;
        });

        // Define custom mode
        ace.define("ace/mode/custom", ["require", "exports", "module", "ace/lib/oop", "ace/mode/text", "ace/mode/custom_highlight_rules"], function(require, exports, module) {
            var oop = require("ace/lib/oop");
            var TextMode = require("ace/mode/text").Mode;
            var CustomHighlightRules = require("ace/mode/custom_highlight_rules").CustomHighlightRules;
            
            var Mode = function() {
                this.HighlightRules = CustomHighlightRules;
            };
            oop.inherits(Mode, TextMode);
            
            (function() {
                this.$id = "ace/mode/custom";
            }).call(Mode.prototype);
            
            exports.Mode = Mode;
        });

        // Define custom completer
        var customCompleter = {
            getCompletions: function(editor, session, pos, prefix, callback) {
                var wordList = {
                    // Keywords
                    'lo': 'load',
                    'fr': 'from',
                    'mo': 'model',
                    'ta': 'task',
                    'ml': 'mlModel',
                    'ty': 'type',
                    'se': 'select',
                    'me': 'metric',
                    'st': 'start',
                    'pa': 'parameter',
                    'na': 'name',
                    'va': 'value',
                    'wh': 'where',
                    'us': 'using',
                    'ru': 'ruleSet',
                    'if': 'if',
                    'th': 'then',
                    'sh': 'show',
                    'pr': 'predict',
                    'cl': 'classification',
                    're': 'regression',
        
                    // ML Models
                    'ran': 'RandomForest',
                    'dec': 'DecisionTree',
                    'svm': 'SVM',
                    'aut': 'AutoML',
        
                    // Metrics
                    'rms': 'rmse',
                    'mae': 'mae',
                    'mse': 'mse',
                    'acc': 'accuracy',
                    'pre': 'precision',
                    'rec': 'recall',
                    'f1': 'f1',
                    'auc': 'auc',
                    'log': 'logloss',
                    'err': 'error',
                    'ham': 'hamming_loss',
                    'jac': 'jaccard',
                    'mat': 'matthews_corrcoef',
                    'zer': 'zero_one_loss',
                    'ave': 'average_precision',
                    'nll': 'neg_log_loss',
                    'nma': 'neg_mean_absolute_error',
                    'nms': 'neg_mean_squared_error',
                    'nml': 'neg_mean_squared_log_error',
                    'nme': 'neg_median_absolute_error',
                    'bal': 'balanced_accuracy',
                    'ami': 'adjusted_mutual_info_score',
                    'ars': 'adjusted_rand_score',
                    'com': 'completeness_score',
                    'exp': 'explained_variance',
                    'fow': 'fowlkes_mallows_score',
                    'hom': 'homogeneity_score',
                    'mis': 'mutual_info_score',
                    'nbs': 'neg_brier_score',
                    'nmi': 'normalized_mutual_info_score',
                    'r2': 'r2_score',
                    'ran': 'rand_score',
                    'vme': 'v_measure_score',
                    'all': 'all'
                };
        
                var completions = [];
                for (var word in wordList) {
                    if (word.indexOf(prefix.toLowerCase()) === 0) {
                        completions.push({
                            caption: wordList[word],
                            value: wordList[word],
                            meta: 'keyword'
                        });
                    }
                }
                
                callback(null, completions);
            }
        };

        var langTools = ace.require("ace/ext/language_tools");
        langTools.addCompleter(customCompleter);
    }
})();