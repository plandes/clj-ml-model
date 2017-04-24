package zensols.weka;

import java.util.Random;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.CostMatrix;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.output.prediction.AbstractOutput;
import weka.core.Instance;
import weka.core.Instances;

public class NoCloneInstancesEvaluationDelegate extends Evaluation {
    private static final Log log = LogFactory.getLog(NoCloneInstancesEvaluationDelegate.class);

    private NoCloneInstancesEvaluation m_eval;

    public NoCloneInstancesEvaluationDelegate(Instances inst, NoCloneInstancesEvaluation eval) throws Exception {
	super(inst);
	m_eval = eval;
    }

    public NoCloneInstancesEvaluationDelegate(Instances inst, CostMatrix costMatrix, NoCloneInstancesEvaluation eval) throws Exception {
	super(inst, costMatrix);
	throw new RuntimeException("This constructor impl not supported");
	//m_eval = eval;
    }

    @SuppressWarnings(value = "unchecked")
    protected void updateStatsForClassifier(double[] predictedDistribution,
					    Instance instance) throws Exception {
	// only logging here, but important to note that the `missing` metric
	// should never be > 0; otherwise, we won't update stats since the
	// label is missing
	if (log.isTraceEnabled()) {
	    StringBuilder bld = new StringBuilder();
	    for (int i = 0; i < predictedDistribution.length; i++) {
		bld.append(predictedDistribution[i] + ":");
	    }
	    log.trace(String.format("prediction dist: %s, correct=%.1f, incorrect=%.1f, missing=%.1f",
				    bld, m_Correct, m_Incorrect, m_MissingClass));
	}

	super.updateStatsForClassifier(predictedDistribution, instance);
    }

    @SuppressWarnings(value = "unchecked")
    public void crossValidateModel(Classifier classifier, Instances data,
				   int numFolds, Random random, Object... forPredictionsPrinting)
	throws Exception {
	if (log.isDebugEnabled()) {
	    log.debug(String.format("classifying instances (no clone) data: %s", data.getClass()));
	}
	// Make a copy of the data we can reorder
	//data = new Instances(data);
	data.randomize(random);
	if (data.classAttribute().isNominal()) {
	    data.stratify(numFolds);
	}

	// We assume that the first element is a
	// weka.classifiers.evaluation.output.prediction.AbstractOutput object
	AbstractOutput classificationOutput = null;
	if (forPredictionsPrinting.length > 0) {
	    // print the header first
	    classificationOutput = (AbstractOutput) forPredictionsPrinting[0];
	    classificationOutput.setHeader(data);
	    classificationOutput.printHeader();
	}

	// Do the folds
	for (int i = 0; i < numFolds; i++) {
	    Instances train = data.trainCV(numFolds, i, random);
	    if (log.isTraceEnabled()) {
		log.trace(String.format("fold: %d; setting priors from trained: <%s>", i, train));
	    }
	    setPriors(train);
	    Classifier copiedClassifier = AbstractClassifier.makeCopy(classifier);
	    copiedClassifier.buildClassifier(train);

	    if (log.isDebugEnabled()) {
		log.debug(String.format("built classifier: %s", copiedClassifier));
	    }

	    Instances test = data.testCV(numFolds, i);
	    if (log.isTraceEnabled()) {
		log.trace(String.format("fold: %d; evaluating with test: <%s>", i, test));
		log.trace(String.format("fold: %d; classifier <%s>", i, copiedClassifier));
	    }

	    double[] preds = evaluateModel(copiedClassifier, test, forPredictionsPrinting);

	    if (log.isTraceEnabled()) {
		log.trace(String.format("fold: %d; finished evaluation: <%s>", i, test));
		log.trace(String.format("fold: %d; classifier <%s>", i, copiedClassifier));
		for (int j = 0; j < preds.length; j++) {
		    log.trace(String.format("prediction: [%d]: %.2f", j, preds[j]));
		}
	    }

	    m_eval.m_trainInstances.add(train);
	    m_eval.m_testInstances.add(test);
	}
	m_NumFolds = numFolds;

	if (classificationOutput != null) {
	    classificationOutput.printFooter();
	}
    }
}
