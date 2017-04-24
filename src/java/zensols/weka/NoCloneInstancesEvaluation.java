package zensols.weka;

import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import weka.classifiers.Evaluation;
import weka.classifiers.CostMatrix;
import weka.core.Instances;

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

    List m_trainInstances;
    List m_testInstances;

    public NoCloneInstancesEvaluation(Instances inst) throws Exception {
	super(inst);
	init();
	m_delegate = new NoCloneInstancesEvaluationDelegate(inst, this);
    }

    public NoCloneInstancesEvaluation(Instances inst, CostMatrix costMatrix) throws Exception {
	super(inst);
	init();
	m_delegate = new NoCloneInstancesEvaluationDelegate(inst, costMatrix, this);
    }

    private void init() {
	m_trainInstances = new java.util.LinkedList();
	m_testInstances = new java.util.LinkedList();
    }

    public List getTrainInstances() {
	return m_trainInstances;
    }

    public List getTestInstances() {
	return m_testInstances;
    }

    public weka.classifiers.evaluation.Evaluation getDelegate() {
	return m_delegate;
    }
}
