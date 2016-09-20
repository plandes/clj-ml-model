/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
package com.zensols.weka;

import clojure.lang.IFn;
import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.WeightedInstancesHandler;

public class IFnClassifier
    extends Classifier 
    implements WeightedInstancesHandler, Sourcable {

    private IFn func;

    public IFnClassifier(IFn func) {
	this.func = func;
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return      the capabilities of this classifier
     */
    public Capabilities getCapabilities() {
	Capabilities result = super.getCapabilities();
	result.disableAll();

	// attributes
	result.enable(Capability.NOMINAL_ATTRIBUTES);
	result.enable(Capability.NUMERIC_ATTRIBUTES);
	result.enable(Capability.DATE_ATTRIBUTES);
	result.enable(Capability.STRING_ATTRIBUTES);
	result.enable(Capability.RELATIONAL_ATTRIBUTES);
	result.enable(Capability.MISSING_VALUES);

	// class
	result.enable(Capability.NOMINAL_CLASS);
	result.enable(Capability.NUMERIC_CLASS);
	result.enable(Capability.DATE_CLASS);
	result.enable(Capability.MISSING_CLASS_VALUES);

	// instances
	result.setMinimumNumberInstances(0);
    
	return result;
    }

    /** no-op **/
    public void buildClassifier(Instances instances) throws Exception {
    }

    public String toSource(String className) throws Exception {
	return "class " + className + ": Clojure function: " + func;
    }

    public double classifyInstance(Instance instance) {
	Number ret = (Number)func.invoke(instance);
	return ret.doubleValue();
    }
}
