package zensols.weka;

import java.util.Random;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.output.prediction.AbstractOutput;
import weka.core.Instances;

public class NoCloneInstancesEvaluationDelegate extends Evaluation {
    private static final Log log = LogFactory.getLog(NoCloneInstancesEvaluationDelegate.class);

    private NoCloneInstancesEvaluation m_eval;

    public NoCloneInstancesEvaluationDelegate(Instances inst, NoCloneInstancesEvaluation eval) throws Exception {
	super(inst);
	m_eval = eval;
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
	    setPriors(train);
	    Classifier copiedClassifier = AbstractClassifier.makeCopy(classifier);
	    copiedClassifier.buildClassifier(train);
	    Instances test = data.testCV(numFolds, i);
	    evaluateModel(copiedClassifier, test, forPredictionsPrinting);

	    m_eval.m_trainInstances.add(train);
	    m_eval.m_testInstances.add(test);
	}
	m_NumFolds = numFolds;

	if (classificationOutput != null) {
	    classificationOutput.printFooter();
	}
    }
}
