"""
    classmetrics(predicted, actual) -> TP, FP, TN, FN, sensitivity, specificity

Returns some metrics i.e. true positives, false positives, true negatives, false negatives, sensitivity and specificity

# Arguments
- 'predicted': What we predicted the labels to be.
- 'actual': What the labels are.
"""
function classmetrics(predicted::AbstractVector{Bool}, actual::AbstractVector{Bool})
    TP = sum(predicted[actual]) / sum(actual)
    FN = sum(.!predicted[actual]) / sum(.!actual)
    FP = sum(predicted[.!actual]) / sum(actual)
    TN = sum(.!predicted[.!actual]) / sum(.!actual)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    @debug "TP = $TP, FP = $FP, TN = $TN , FN = $FN"
    @debug "TPR= $sensitivity = $TP, TNR = $specificity"
    return TP, FP, TN, FN, sensitivity, specificity
end

"""
    confusionmatrix(predictlabels, labelstest) -> confusionmatrix

A confusion matrix is a summary of prediction results on a classification problem.
The number of correct and incorrect predictions are summarized with count values and broken down by each class. This is the key to the confusion matrix.
The confusion matrix shows the ways in which your classification model is confused when it makes predictions.

# Arguments
- 'predictlabel': What we predicted the labels to be.
- 'labelstest': What the labels are.
"""
function confusionmatrix(predictlabels::AbstractVector, labelstest::AbstractVector, normalised::Bool = true)
    labels = sort(unique(labelstest)) # Labels need to be sortable
    confusionmatrix(predictlabels, labelstest, labels, normalised)
end

function confusionmatrix(predictlabels::AbstractVector, labelstest::AbstractVector, labels::AbstractVector, normalised::Bool = true)
    N = length(labels)
    confusionmat = zeros(N,N)
    for (i, label) in enumerate(labels)
        sel = labelstest .== label
        predictions = predictlabels[sel]
        Nlabels = sum(sel)
        for (j, predlabel) in enumerate(labels)
            if normalised
                confusionmat[i,j] = sum(predictions .== predlabel)/Nlabels
            else
                confusionmat[i,j] = sum(predictions .== predlabel)
            end
        end
    end
    return confusionmat
end

# Normalize a confusion matrix
function normalizeconfusionmatrix!(confusionmat::Matrix)
    n, m = size(confusionmat)
    @assert n == m
    rowsum = sum(confusionmat, dims=2)
    for i = 1:n
        rowsum[i] == 0 && continue
        confusionmat[i, :] ./= rowsum[i]
    end
    return confusionmat
end

"""
    cohenkappa(predictlabels, labelstest) -> kappa

A strong multiclass classification metric when there is significant class imbalance.
Example, if there was a class with high prevalance, it may have a high precision and recall,
so an F1 measure will return a high score, whereas cohen's kappa score will rightly return a low score.

Closer to 1 the better.

http://standardwisdom.com/softwarejournal/2013/04/comparing-kappa-statistic-to-the-f1-measure/

# Arguments
- 'predictlabel': What we predicted the labels to be.
- 'labelstest': What the labels are.
"""
function cohenkappa(predictlabels::AbstractVector, labelstest::AbstractVector)
    confusionmat = confusionmatrix(predictlabels, labelstest, false)
    cohenkappa(confusionmat)
end

function cohenkappa(confusionmat::AbstractMatrix)
    accuracy = tr(confusionmat) / sum(confusionmat) # observed accuracy
    expected = tr((sum(confusionmat, dims=2)*sum(confusionmat, dims=1))/sum(confusionmat)^2) # expected accuracy
    kappa = (accuracy - expected) / (1.0 - expected)
end