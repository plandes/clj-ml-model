(ns ^{:doc "Wraps the Weka Java API.  This is probably the wrong library to use
for most uses.  Instead take a look at [[zensols.model.eval-classifier]]
and [[zensols.model.execute-classifier]]."
      :author "Paul Landes"}
    zensols.model.weka
  (:use [clojure.pprint :only [pprint]])
  (:require [clojure.tools.logging :as log]
            [clojure.string :as str])
  (:import (weka.core Attribute Instance Instances FastVector))
  (:import (weka.core.converters ArffLoader ConverterUtils$DataSink))
  (:import (weka.classifiers Classifier Evaluation))
  (:import (weka.filters Filter))
  (:import (weka.filters.unsupervised.attribute Remove)))

(def ^:dynamic *classifiers*
  "An (incomplete) set of Weka classifiers keyed by their speed, type or
  singleton by name.

  * **fast** train quickly
  * **slow** train slowly
  * **really-slow** train very very slowly
  * **lazy** lazy category
  * **meta** meta classifiers (i.e. boosting)
  * **tree** tree based classifiers (typically train quickly)

  The singleton classifiers is a list like the others but have only a single
  element of the class.  They include: `zeror`, `svm`, `j48`, `random-forest`,
  `naivebays`, `logit`, `logitboost`, `smo`, `kstar`."
 {:fast '("weka.classifiers.trees.J48"
           "weka.classifiers.rules.ZeroR"
           "weka.classifiers.bayes.NaiveBayes"
           "weka.classifiers.rules.Ridor"
           "weka.classifiers.rules.JRip"
           "weka.classifiers.rules.PART"
           "weka.classifiers.rules.ConjunctiveRule"
           "weka.classifiers.rules.DecisionTable"
           "weka.classifiers.rules.NNge"
           "weka.classifiers.bayes.BayesNet")
   :slow '("weka.classifiers.functions.SimpleLogistic"
           "weka.classifiers.functions.SMO"
           "weka.classifiers.functions.LibSVM"
           "weka.classifiers.rules.DTNB")
   :really-slow '("weka.classifiers.functions.MultilayerPerceptron"
                  "weka.classifiers.functions.Logistic")
   :lazy '("weka.classifiers.lazy.KStar"
           "weka.classifiers.lazy.IB1"
           "weka.classifiers.lazy.IBk"
           "weka.classifiers.lazy.LWL")
   :meta '("weka.classifiers.meta.AdaBoostM1"
           "weka.classifiers.meta.LogitBoost"
           "weka.classifiers.meta.Stacking"
           "weka.classifiers.meta.Vote"
           "weka.classifiers.trees.DecisionStump"
           "weka.classifiers.meta.Grading"
           "weka.classifiers.meta.Bagging")
   :tree '("weka.classifiers.misc.HyperPipes"
           "weka.classifiers.trees.RandomTree"
           "weka.classifiers.trees.RandomForest"
           "weka.classifiers.trees.NBTree"
           "weka.classifiers.trees.REPTree")
   :zeror '("weka.classifiers.rules.ZeroR")
   :svm '("weka.classifiers.functions.LibSVM")
   :j48 '("weka.classifiers.trees.J48")
   :random-forest '("weka.classifiers.trees.RandomForest")
   :naivebayes '("weka.classifiers.bayes.NaiveBayes")
   :logit '("weka.classifiers.functions.SimpleLogistic")
   :logitboost '("weka.classifiers.meta.LogitBoost")
   :smo '("weka.classifiers.functions.SMO")
   :kstar '("weka.classifiers.lazy.KStar")})

(def ^:dynamic *missing-values-ok*
  "Whether missing the classifier can handle missing values, otherwise an
  exception is thrown for missing values."
  false)

(defn make-classifiers
  "Make classifiers from either a key in [[*classifiers*]] or an instance of
  `weka.classifiers.Classifier` (meaning an already constructed instance).  All
  classifiers are returned for the 0-arg option."
  ([]
   (map #(.newInstance (Class/forName %))
        (distinct (reduce concat (map second *classifiers*)))))
  ([set-name-or-instance]
   (log/debugf "making classifiers: %s" set-name-or-instance)
   (if (nil? set-name-or-instance)
     (make-classifiers)
     (if (instance? weka.classifiers.Classifier set-name-or-instance)
       (list set-name-or-instance)
       (map (fn [cl-name]
              (.newInstance (Class/forName cl-name)))
            (get *classifiers* set-name-or-instance))))))

(defn create-attrib
  "Create a Weka Attribute instance with **att-name**.

  **type** is the type of attribute, which can be `string`, `boolean`,
  `numeric`, or a sequence of strings representing possible enumeration
  values (nominals in Weka speak)."
  [att-name type]
  (log/debugf "create-attrib: att-name=%s, type=%s" att-name (pr-str type))
  (if (sequential? type)
    (let [fv (FastVector. (count type))]
      (doseq [label type]
        (if (nil? label)
          (throw (ex-info (format "Missing label for attribute: %s" att-name)
                          {:attribute-name att-name
                           :fast-vector fv})))
        (log/debugf "creating attribute: name=%s, label=%s" att-name label)
        (.addElement fv (name label)))
      (Attribute. att-name fv))
    (case type
      string ((fn [^String aname ^FastVector v]
                (Attribute. aname v))
              att-name nil)
      boolean (Attribute. att-name)
      numeric (Attribute. att-name))))

(defn- classifier-label-attrib [label]
  (if label
    (create-attrib (name (first label)) (second label))))

(defn- classifier-attribs [feature-metas]
  (apply merge
         (map (fn [[key type]]
                {key (create-attrib (name key) type)})
              feature-metas)))

(defn- create-feature-attrib-set
  "Create a map containing an `:attrib` `weka.core.Attribute`, `:key` of the
  key in feature-metas and `:type` as the type given in **attribs**."
  [attribs feature-metas]
  (map (fn [key]
         {:attrib (get attribs key)
          :key key
          :type (get feature-metas key)})
       (keys attribs)))

(defn value-for-instance
  "Return a Java variable that plays nicely with the Weka framework.  If no
  **type** is given it tries to determine the type on its own.

  * **val** is a Java primitive (wrapper)
  * **type** if given, is the type of **val** (see [[create-attrib]])"
  ([val]
   (cond (= Boolean (type val)) (double (if val 1 0))
         (number? val) (double val)
         true (String/valueOf val)))
  ([type val]
   (if-not (nil? val)
     (if (sequential? type)
       val
       (case type
         numeric (if (string? val)
                   (throw (ex-info (format "Value is a string: %s" val)
                                   {:value val}))
                   (double val))
         boolean (double (if val 1 0))
         string (if (nil? val) "" val))))))

(defn attribute-by-name
  "Return a `weka.core.Attribute` instance by name from a
  `weka.core.Instances`."
  [instances name]
  (first (filter #(not (nil? %))
                 (map (fn [att]
                        (if (= (.name att) name)
                          att))
                      (enumeration-seq (.enumerateAttributes instances))))))

(defn attributes-for-instances
  "Return a map with **:name** and **:type** for each attribute in an
  `weka.core.Instances`."
  [insts]
  (->> (enumeration-seq (.enumerateAttributes insts))
       (map (fn [attrib]
              {:name (.name attrib)
               :type (case (.type attrib)
                       Attribute/DATE 'date
                       Attribute/NOMINAL 'nominal
                       Attribute/NUMERIC 'numeric
                       Attribute/STRING 'string
                       nil)}))
       (sort (fn [a b]
               (compare (:name a) (:name b))))))

(defn- new-instances
  "Generate new a map that has a new Instances instance along with all the
  metadata.

  * **inst-name** used to identify the model data set

  * **feature-metas** a map of key/value pairs describing the features (they
  become `weka.core.Attribute`s) where the values are described as types
  in [[create-attrib]]

  * **class-feature-meta** just like a (single) **feature-metas** but describes
  the class

  * **add-label-spot?** whether to add an additional attribute for the class if
  the class isn't given in cases where it's populated by the caller"
  [inst-name feature-metas class-feature-meta add-label-spot?]
  (let [label-attrib (classifier-label-attrib class-feature-meta)
        attribs (create-feature-attrib-set (classifier-attribs feature-metas)
                                           feature-metas)
        attrib-count (+ (if add-label-spot? 1 0) (count attribs))
        attrib-vec (FastVector. attrib-count)]
    (doseq [attrib attribs]
      (.addElement attrib-vec (:attrib attrib)))
    (if label-attrib (.addElement attrib-vec label-attrib))
    (let [insts (Instances. inst-name attrib-vec 0)]
      (if label-attrib (.setClassIndex insts (.index label-attrib)))
      {:instances insts
       :attribs attribs
       :attrib-count attrib-count
       :label-attrib label-attrib})))

(defn- populate-instance
  "Populate an `Instance` with data from a map for a set of attributes.

  * **inst** an `weka.core.Instance` that will be populated (this is a row in
  an `Instances`)

  * **data-map** a map where the keys (Clojure keys) take the same name as the
  attributes and the values are the literal values that will be set on **inst**

  * **attrib-metas** list of maps with each map having:
      * `:attrib` => `weka.core.Attribute`
      * `:key` => keyword attribute name
      * `:type` => same type as described in [[create-attrib]]"
  [inst data-map attrib-metas]
  (doseq [attrib attrib-metas]
    (let [atinst (:attrib attrib)
          type (:type attrib)
          val (get data-map (:key attrib))
          atval (try
                  (value-for-instance type val)
                  (catch Exception e
                    (throw (ex-info (str "Can't get attribute value for instance: " e)
                                    {:type type
                                     :attribute atinst
                                     :value val}))))]
      (log/tracef "feature attrib: %s, type: %s, val: %s" attrib (pr-str type) val)
      (if (and ;(or (= type 'string) (sequential? type)) ; string or nom
           (and (nil? val) (not *missing-values-ok*)))
        (throw (ex-info (format "No value for attribute %s" atinst)
                        {:attribute atinst :type type :val val
                         :data-map data-map}))
        (try
          (if atval (.setValue inst atinst atval))
          (catch Exception e
            (throw (ex-info (format "Can't set instance feature: %s=%s: %s"
                                    atinst atval e)
                            {:type type
                             :attribute atinst
                             :value val}))))))))

(defn- create-instances
  "Create a new `weka.core.Instances` instance.

  * **inst-name** used to identify the model data set

  * **feature-sets** a sequence of maps with each map having key/value pairs of
  the features of the model to be populated in the returned
  `weka.core.Instances`

  * **attrib-metas** list of maps with each map having:
      * `:attrib` => `weka.core.Attribute`
      * `:key` => keyword attribute name
      * `:type` => same type as described in [[create-attrib]]

  * **class-feature-meta** just like a (single) **feature-metas** but describes
  the class

  * **add-label-spot?** whether to add an additional attribute for the class if
  the class isn't given in cases where it's populated by the caller"
  [inst-name feature-sets attrib-metas class-feature-meta add-label-spot?]
  (let [{:keys [instances attribs label-attrib attrib-count]}
        (new-instances inst-name attrib-metas class-feature-meta add-label-spot?)
        insts instances]
    (if label-attrib (.setClassIndex insts (.index label-attrib)))
    (doseq [feature-set feature-sets]
      (log/debugf "feature set: <%s>" (pr-str feature-set))
      (log/debugf "classifier label def: %s" class-feature-meta)
      (let [features (select-keys feature-set (keys attrib-metas))
            ;; this has to be a string (not even boolean) since some
            ;; classifiers (J48) can't handle numeric labels since they don't
            ;; handle prediction--make these nominitives even for booleans
            label-type (second class-feature-meta)
            label-val (get feature-set (first class-feature-meta))]
        (log/debugf "label val: %s" label-val)
        (let [inst (Instance. attrib-count)]
          ;; not set for when used for predictions/exercising the model
          (when (not (nil? label-val))
            (let [vfi (value-for-instance label-type label-val)]
              (log/debugf "val for inst val: %s->%s" label-attrib vfi)
              (try
                (.setValue inst label-attrib vfi)
                (catch Exception e
                  (throw (ex-info (str "Can't set instances value: " e)
                                  {:label-attrib label-attrib
                                   :label-val label-val
                                   :value vfi}))))))
          (populate-instance inst features attribs)
          (.setDataset inst insts)
          (.add insts inst))))
    insts))

(defn map-merge-instances
  "Merges two sets of Instances together. The resulting set will have all the
  attributes of the first set plus all the attributes of the second set.

  * **src** a `weka.core.Instances` containing the first part to be merged

  * **attrib-metas** list of maps with each map having:
      * `:attrib` => `weka.core.Attribute`
      * `:key` => keyword attribute name
      * `:type` => same type as described in [[create-attrib]]

  * **instance-val-fn** a function called with a single unique identifier used
  to add to the result, which makes the second part of the combined merged
  result"
  [src attrib-metas instance-val-fn]
  (log/debugf "merge with attribs: %s" attrib-metas)
  (let [features-sets (map (fn [idx]
                             (instance-val-fn (.instance src idx)))
                           (range (.numInstances src)))
        dst (create-instances "merged" features-sets attrib-metas nil false)
        class-attrib (.classAttribute src)
        merged (weka.core.Instances/mergeInstances src dst)]
    (if class-attrib (.setClass merged class-attrib))
    (log/debugf "class: %s" class-attrib)
    merged))

(defn- find-missing-nominals
  "Some classifiers can't deal with missing nominals.  Weka gives a nasty
  exception if this happens, so detect this (pretty common) mistake before it
  gets into the Weka layer and report what's missing.

  * **feature-sets** a sequence of maps with each map having key/value pairs of
  the features of the model to be populated in the returned
  `weka.core.Instances`

  * **feature-metas** a map of key/value pairs describing the features (they
  become attributes) where the values are described as types
  in [[create-attrib]]"
  [feature-sets feature-metas]
  (doall
   (filter
    #(not (empty? %))
    (apply concat
           (map (fn [[key val]]
                  (when (sequential? val)
                    (let [nom (set val)]
                      (filter
                       identity
                       (doall
                        (map (fn [feature]
                               (let [val (get feature key)
                                     found-in? (contains? nom val)]
                                 (log/tracef "key=%s, val=%s, found in: %s"
                                             key val found-in?)
                                 (if (not found-in?)
                                   {:key key
                                    :value val
                                    :feature feature})))
                             feature-sets))))))
                feature-metas)))))

(defn clone-instances
  "Return a deep clone of **inst**, optionally with a specific training and
  test set.

  Parameters
  ----------
  * **inst** an instance of `weka.core.Instances` (note the whole dataset)

  Keys
  ----
  * **train-fn** a function that takes the following arguments: an
  `weka.core.Instances` created for the training set, number of folds, the fold
  number and a `java.util.Random` to pass to the Weka layer to shuffle the
  dataset

  * **test-fn** just like **train-fn** but used to create the test data set and
  it doesn't take the `java.util.Random` instance"
  [inst & {:keys [train-fn test-fn randomize-fn]}]
  (let [train-state (atom nil)]
   (proxy [Instances] [inst]
     (randomize [rand]
       (log/infof "randominzing: %s with fn=%s" rand randomize-fn)
       (if randomize-fn (randomize-fn rand)))
     (trainCV [folds fold rand]
       (log/debugf "folds: %d, fold no: %d" folds fold)
       (let [res (proxy-super trainCV folds fold rand)]
         (if train-fn
           (train-fn res train-state inst folds fold)
           res)))
     (testCV [folds fold]
       (log/debugf "folds: %d, fold no: %d" folds fold)
       (let [res (proxy-super testCV folds fold)]
         (if test-fn
           (test-fn res train-state inst folds fold)
           res))))))

(defn instances
  "Create a new `weka.core.Instances` instance.

  * **inst-name** used to identify the model data set

  * **feature-sets** a sequence of maps with each map having key/value pairs of
  the features of the model to be populated in the returned
  `weka.core.Instances`

  * **feature-metas** a map of key/value pairs describing the features (they
  become `weka.core.Attribute`s) where the values are `string`, `boolean`,
  `numeric`, or a sequence of strings representing possible enumeration
  values (nominals in Weka speak)

  * **class-feature-meta** just like a (single) **feature-metas** but describes
  the class"
  [inst-name feature-sets feature-metas class-feature-meta]
  (log/debugf "create %d instances" (count feature-sets))
  (let [missing-noms (find-missing-nominals feature-sets feature-metas)]
    (if (and (not *missing-values-ok*)
             (not (empty? missing-noms)))
      (throw (ex-info "Missing nominal (only first reported)"
                      (first missing-noms)))))
  (let [inst (create-instances inst-name feature-sets
                               feature-metas class-feature-meta true)]
    (clone-instances inst)))

(defn- index-for-attribute
  "Return an `weka.core.Attribute` in **inst** (`weka.core.Instances`)
  by (string) name."
  [inst name]
  (let [attrib (attribute-by-name inst name)]
    (if attrib (.index attrib))))

(defn remove-attributes
  "Remove a set of attributes from **inst** (`weka.core.Instances`) by
  string (string) name."
  [inst attrib-names]
  (let [remove-filter (Remove.)
        indexes (doall (map #(index-for-attribute inst %) attrib-names))]
    (if-not (empty? (filter nil? indexes))
      (throw (ex-info (format "indexes: %s=>%s"
                              (str/join ", " attrib-names)
                              (str/join ", " indexes))
                      {:attrib-names attrib-names
                       :indexes indexes})))
    (.setAttributeIndicesArray remove-filter (into-array Integer/TYPE indexes))
    (.setInputFormat remove-filter inst)
    (Filter/useFilter inst remove-filter)))
