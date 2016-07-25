package zensols.weka;

import java.util.List;
import java.util.Random;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Range;

/**
 * <p>Exactly like the parent class but don't clone the instances on cross
 * validation.</p>
 *
 * <p>The Weka implementation of {@link Evaluation} clones the {@link Instances}
 * instance passed in to {@link #crossValidateModel}, which we don't want to
 * do.  We don't want to do this since we extend {@link Instances} in
 * <tt>zensols.model.weka.clone-instances</tt> for a two pass cross fold
 * validation.</p>
 *
 * @author Paul Landes
 */
public class NoCloneInstancesEvaluation extends Evaluation {
    private static final Log log = LogFactory.getLog(NoCloneInstancesEvaluation.class);

    private List trainInstances;
    private List testInstances;

    public NoCloneInstancesEvaluation(Instances inst) throws Exception {
	super(inst);
	trainInstances = new java.util.LinkedList();
	testInstances = new java.util.LinkedList();
    }

    public List getTrainInstances() {
	return trainInstances;
    }

    public List getTestInstances() {
	return testInstances;
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

	// We assume that the first element is a StringBuffer, the second a Range
	// (attributes
	// to output) and the third a Boolean (whether or not to output a
	// distribution instead
	// of just a classification)
	if (forPredictionsPrinting.length > 0) {
	    // print the header first
	    StringBuffer buff = (StringBuffer) forPredictionsPrinting[0];
	    Range attsToOutput = (Range) forPredictionsPrinting[1];
	    boolean printDist = ((Boolean) forPredictionsPrinting[2]).booleanValue();
	    printClassificationsHeader(data, attsToOutput, printDist, buff);
	}

	// Do the folds
	for (int i = 0; i < numFolds; i++) {
	    Instances train = data.trainCV(numFolds, i, random);
	    setPriors(train);
	    Classifier copiedClassifier = Classifier.makeCopy(classifier);
	    copiedClassifier.buildClassifier(train);
	    Instances test = data.testCV(numFolds, i);
	    evaluateModel(copiedClassifier, test, forPredictionsPrinting);

	    trainInstances.add(train);
	    testInstances.add(test);
	}
	m_NumFolds = numFolds;
    }
}
