;; Written by Paul Landes -- Dec 2014

(ns ^{:doc
"A utility library that wraps Weka library.  This library works
    with [[zensols.model.weka]] do the following:
  * Cross validate models
  * Manage and sort results (i.e. cross validations)
  * Train models
  * Read/write ARFF files

This namspace uses the [resource
location](https://github.com/plandes/clj-actioncli#resource-location) system to
configure the location of files and output analysis files.  For more
information about the configuration specifics see [[model-read-resource]]
and [[analysis-dir]], which both
use [resource-path](https://plandes.github.io/clj-actioncli/codox/zensols.actioncli.resource.html#var-resource-path).

You probably don't want to use this library directly.  Please look
at [[zensols.model.eval-classifier]] and [[zensols.model.execute-classifier]]."
      :author "Paul Landes"}
    zensols.model.classifier
  (:use [clojure.java.io :as io :only [input-stream output-stream]])
  (:use [clojure.pprint :only [pprint]])
  (:use [clj-excel.core :as excel])
  (:require [clojure.tools.logging :as log]
            [clojure.string :as str])
  (:import [java.io File])
  (:import (zensols.weka NoCloneInstancesEvaluation))
  (:import (weka.classifiers Classifier Evaluation)
           (weka.filters.unsupervised.attribute Remove)
           (weka.filters.supervised.attribute AddClassification)
           (weka.core.converters ArffLoader ConverterUtils$DataSink)
           (weka.filters Filter)
           (weka.core Utils Instances))
  (:require [zensols.actioncli.resource :as res])
  (:require [zensols.model.weka :as weka]))

(def ^:private zero-arg-arr (into-array String []))

(def ^:dynamic *arff-file*
  "File to read or write from for any operation regarding file system access to
  a/the ARFF file(s)."
  nil)

(def ^:dynamic *class-feature-meta*
  "The class feature metadata (see [[zensols.model.weka/create-attrib]])."
  nil)

(def ^:dynamic *output-class-feature-meta*
  "Default attribute name for the predicted label."
  "classification")

(def ^:dynamic *classifier-class*
  "Class name for the classifier used.  This defaults to J48."
  "weka.classifiers.trees.J48")

(def ^:dynamic *cross-val-fns*
  "If this is non-`nil` then two pass validation is used.  This is a map with
  the following keys:

  * **:train-fn** a function that is called during training for each fold to
  *stitch in* partial feature-sets to get better results; almost always set
  to [[zensols.model.eval-classifier/two-pass-train-instances]]

  * **:test-fn** just like **:train-fn** but called during testing; almost
  always set to [[zensols.model.eval-classifier/two-pass-train-instances]]

  See [[zensols.model.eval-classifier/*two-pass-config*]]"
  nil)

(def ^:dynamic *best-result-criteria*
  "Key used to sort results by their most optimal performance statistic.  Valid
  values are: `:accuracy`, `:wprecision`, `:wrecall`, `:wfmeasure`,
  `:kappa`, `:rmse`"
  :wfmeasure)

(def ^:dynamic *create-classifier-fn*
  "Function used to create a classifier.  Takes as input a
  `weka.core.Instances`."
  (fn [_]
    (Utils/forName Classifier *classifier-class* zero-arg-arr)))

(def ^:dynamic *cross-fold-count*
  "The default number of folds to use during cross fold
  validation (see [[cmpile-results]])."
  10)

(def ^:dynamic *operation-write-instance-fns*
  "A map with valus of functions that are called that return a `java.util.File`
  for an operation represented by the respective key.  An ARFF file is created
  at the file location.  The keys are one of:

  * **:train-classifier** called when the classifier is training a model
  * **:test-classifier** called when the classifier is testing a model"
  nil)

(def ^:dynamic excel-results-precision
  "An integer specifying the length of the mantissa when creating the results
  spreadsheet in [[excel-results]]."
  5)

(defn initialize
  "Initialize model resource locations.

  This needs the system property `clj.nlp.parse.model` set to a directory that
  has the POS tagger model `english-left3words-distsim.tagger`(or whatever
  you configure in [[zensols.nlparse.stanford/create-context]]) in a directory
  called `pos`.

  See the [source documentation](https://github.com/plandes/clj-nlp-parse) for
  more information."
  []
  (log/debug "initializing")
  (res/register-resource :model :system-property "model")
  (res/register-resource :model-write :pre-path :model :system-file "zensols")
  (res/register-resource :model-read :pre-path :model :system-file "zensols")
  (res/register-resource :analysis-report :system-file
                         (-> (System/getProperty "user.home")
                             (io/file "Desktop")
                             .getAbsolutePath)))

(defn analysis-report-resource
  "Return the model directory on the file system as defined by the
  `:analysis-report`.  See namespace documentation on how to configure."
  []
  (res/resource-path :analysis-report))

(defn model-read-resource
  "Return a file pointing to model with `name` using the the `:model-read`
  resource path (see [[zensols.actioncli.resource/resource-path]])."
  [name]
  (res/resource-path :model-read (format "%s.dat" name)))

(defn model-write-resource
  "Return a file pointing to model with `name` using the the `:model-write`
  resource path (see [[zensols.actioncli.resource/resource-path]])."
  [name]
  (res/resource-path :model-write (format "%s.dat" name)))

(defn model-exists?
  "Return whether a the model exists with `name`.

  See [[model-read-resource]]."
  [name]
  (.exists (model-read-resource name)))

(defn read-model
  "Get a saved model (classifier and attributes used).  If **name** is a
  string, use [[model-read-resource]] to calculate the file name.  Otherwise,
  it should be a file of where the model exists.

  See [[model-read-resource]].

  Keys
  ----
  * **:fail-if-not-exists?** if `true` then throw an exception if the model
  file is missing"
  [name & {:keys [fail-if-not-exists?]
           :or {fail-if-not-exists? true}}]
  (let [res (if (instance? File name)
              name
              (model-read-resource name))
        file-res? (instance? File res)
        exists? (and file-res? (.exists res))]
    (if (and fail-if-not-exists? file-res? (not exists?))
      (throw (ex-info (format "no model file found: %s" res)
                      {:file res})))
    (if (and file-res? (not exists?))
      (log/infof "no model resource found %s" res)
      (do
        (log/infof "reading model from %s" res)
        (with-open [in (input-stream res)]
          (let [in-obj (java.io.ObjectInputStream. in)]
            (.readObject in-obj)))))))

(defn write-model
  "Get a saved model (classifier and attributes used).  If **name** is a
  string, use [[model-write-resource]] to calculate the file name.  Otherwise,
  it should be a file of where to write the model.

  See [[model-read-resource]]"
  [name model]
  (let [file (if (instance? File name)
              name
              (model-read-resource name))]
    (.mkdirs (.getParentFile file))
    (with-open [out (output-stream file)]
      (let [out-obj (java.io.ObjectOutputStream. out)]
        (.writeObject out-obj model)))
    (log/infof "saved model to %s" file)
    model))

(defn read-arff
  "Return a `weka.core.Instances` from an ARFF file."
  [input-file]
  (log/infof "reading ARFF file: %s" input-file)
  (with-open [reader (io/reader input-file)]
    (Instances. reader)))

(defn write-arff
  "Write a `weka.core.Instances` to an ARFF file and return that file."
  [instances]
  (log/infof "writing ARFF file: %s" *arff-file*)
  (ConverterUtils$DataSink/write
   (.getAbsolutePath *arff-file*) instances)
  *arff-file*)

(defn- create-classifier
  "Create a classifier instance."
  [data]
  (apply *create-classifier-fn* (list data)))

(defn- set-classify-attrib
  "Set the attribute to classify on DATA."
  [data]
  (let [attrib (.attribute data *class-feature-meta*)]
    (.setClass data attrib)
    attrib))

(def ^:dynamic *get-data-fn*
  "A function that generates a `weka.core.Instances` for cross validation,
  training, etc."
  (fn []
    (let [loader (ArffLoader.)]
      (.setFile loader *arff-file*)
      (let [data (.getDataSet loader)]
        (set-classify-attrib data)
        data))))

(defn- get-data
  "Get the ARFF in memory (Instance) data structure."
  []
  (apply *get-data-fn* nil))

(def ^:dynamic *rand-fn*
  (fn [] (java.util.Random. (System/currentTimeMillis))))

(defn- cross-validate
  "Invoke the Weka wrapper to cross validate.

  In the Weka layer we proxy out a class that lets us do a two pass cross
  validation so here we use our
  overriden [[zensols.weka.NoCloneInstancesEvaluation]] to execute teh
  validation."
  ([folds insts classifier]
   (log/debugf "cross validate instances: no-clone: %s, weka: %s"
               *cross-val-fns* (.getClass insts))
   (let [two-pass-validation? (not (nil? *cross-val-fns*))
         eval (if two-pass-validation?
                (NoCloneInstancesEvaluation. insts)
                (Evaluation. insts))]
     ;; docs say that deep clone is performed on the classifier so it should be
     ;; reusable after evaluation
     (.crossValidateModel eval classifier insts folds (*rand-fn*) zero-arg-arr)
     (merge {:eval eval
             :train-total (.numInstances insts)}
            (if two-pass-validation?
              {:attribs (->> (weka/attributes-for-instances
                              (-> eval (.getTrainInstances) (.get 0)))
                             (map :name))}))))
  ([folds]
   (let [raw-insts (get-data)
         insts (if *cross-val-fns*
                 (apply weka/clone-instances raw-insts
                        (apply concat (into () *cross-val-fns*)))
                 raw-insts)]
     (cross-validate folds insts (create-classifier insts)))))

(defn filter-attribute-data
  "Create a filtered data set (`weka.core.Instances`) from unfiltered Instances.
  Paramater **attributes** is a set of string attribute names."
  [unfiltered attributes]
  (if-not attributes
    unfiltered
    (let [filter (Remove.)]
      (log/debugf "attributes: %s, insts: %s"
                  (str/join ", " attributes)
                  (type unfiltered))
      (letfn [(attrib-by-name [aname]
                (or (.attribute unfiltered aname)
                    (throw (ex-info (str "Unknown attribute: " aname)
                                    {:name aname}))))]
        (.setInvertSelection filter true)
        (.setAttributeIndicesArray
         filter
         (int-array
          (map #(.index (attrib-by-name %))
               (concat (if *class-feature-meta* [*class-feature-meta*])
                       attributes))))
        (.setInputFormat filter unfiltered)
        (let [data (Filter/useFilter unfiltered filter)]
          (set-classify-attrib data)
          data)))))

(defn- cross-validate-evaluation
  "Perform a cross validation using **classifier** on **data** Instances.
  Paramater **attributes** is a set of string attribute names."
  [classifier data attributes]
  (let [prev-get-data *get-data-fn*]
    (letfn [(class-fn [data]
              classifier)
            (get-data []
              (filter-attribute-data data attributes))]
      (binding [*create-classifier-fn* class-fn
                *get-data-fn* get-data]
        (cross-validate *cross-fold-count*)))))

(defn- eval-to-results [eval attribs instances-trained classifier]
  {:eval eval
   :feature-metas attribs
   :classifier classifier
   :classify-attrib (keyword *class-feature-meta*)

     ;;; evaluation results
   ;; basic stats
   :instances-total (.numInstances eval)
   :instances-correct (.correct eval)
   :instances-incorrct (.incorrect eval)
   :instances-trained instances-trained

   ;; metrics
   :accuracy (.pctCorrect eval)
   :kappa (.kappa eval)
   :rmse (.errorRate eval)
   :wprecision (.weightedPrecision eval)
   :wrecall (.weightedRecall eval)
   :wfmeasure (.weightedFMeasure eval)})

(defn cross-validate-tests
  "Run the cross validation for **classifier** and **attributes** (symbol
  set)."
  [classifier attributes]
  (log/infof "cross validate tests with classifier %s on %s"
             (.getName (.getClass classifier))
             (if attributes
               (str/join ", " attributes)
               "none"))
  (let [data (get-data)
        _ (log/infof "cross validate on %d instances" (.numInstances data))
        {:keys [eval attribs train-total] :as cve}
        (cross-validate-evaluation classifier data attributes)]
    (merge (select-keys cve [train-total])
           (eval-to-results eval (or attribs attributes '("none"))
                            (/ (.numInstances eval) *cross-fold-count*)
                            classifier))))

(defn train-classifier
  "Train **classifier** (`weka.classifiers.Classifier`)."
  [classifier attributes]
  (log/infof "training classifer %s on %s"
             (.getName (.getClass classifier))
             (if attributes
               (str/join ", " attributes)
               "none"))
  (let [raw-data (get-data)
        _ (log/infof "training on %d instances" (.numInstances raw-data))
        train-data (filter-attribute-data raw-data attributes)
        arff-file (get *operation-write-instance-fns* :train-classifier)]
    (if arff-file
      (binding [*arff-file* arff-file]
        (write-arff train-data)))
    (.buildClassifier classifier train-data)
    classifier))

(defn test-classifier
  "Test/evaluate **classifier** (`weka.classifiers.Classifier`)."
  [classifier attributes train-data test-data]
  (log/infof "testing classifer %s on %s"
             (.getName (.getClass classifier))
             (str/join ", " attributes))
  (let [train-data (filter-attribute-data train-data attributes)
        test-data (->> (filter-attribute-data test-data attributes)
                       weka/clone-instances)
        _ (log/infof "testing on %d instances" (.numInstances test-data))
        eval (Evaluation. train-data)
        arff-file (get *operation-write-instance-fns* :test-classifier)]
    (if arff-file
      (binding [*arff-file* arff-file]
        (write-arff test-data)))
    (.evaluateModel eval classifier test-data zero-arg-arr)
    eval))

(defn train-test-classifier [classifier feature-meta-sets
                             train-instances test-instances]
  (binding [*get-data-fn* #(identity train-instances)]
    (->> feature-meta-sets
         (map #(map name %))
         (map (fn [attribs]
                (log/debugf "classifier: %s, attribs: %s"
                            classifier (pr-str attribs))
                (train-classifier classifier attribs)
                (-> (test-classifier classifier attribs
                                     train-instances test-instances)
                    (eval-to-results attribs (.numInstances train-instances)
                                     classifier)
                    (assoc :train-total (.numInstances train-instances)
                           :test-total (.numInstances test-instances)))))
         doall)))

(defn classify-instance
  "Make predictions for all instances.

  * **classifier** instance of `weka.classifiers.Classifier`

  * **unlabeled** contains feature set data with an empty class label as a
  `weka.core.Instances`

  * **return-keys** what data to return
        * **:label** the classified label
        * **:distributions** the probability distribution over the label"
  [classifier unlabeled return-keys]
  (log/debugf "classify instance: class index: %d: %s"
              (.classIndex unlabeled) (.classAttribute unlabeled))
  (log/debugf "return keys: %s" return-keys)
  (log/tracef "unlabeled: %s" unlabeled)
  (map (fn [idx]
         (let [unlabeled-inst (.instance unlabeled idx)
               label (if (:label return-keys)
                       (.classifyInstance classifier unlabeled-inst))
               label-val (if label
                           (.value (.classAttribute unlabeled) label))
               dists (if (:distributions return-keys)
                       (apply
                        merge
                        (map (fn [dist attrib-name]
                               {attrib-name dist})
                             (.distributionForInstance classifier unlabeled-inst)
                             (enumeration-seq
                              (.enumerateValues
                               (.classAttribute unlabeled-inst))))))]
           (log/debugf "label: %s (%s)" label label-val)
           (log/tracef "dists: %s" dists)
           {:label label-val
            :distributions dists}))
       (range (.numInstances unlabeled))))

(defn- sort-results [results]
  (sort #(compare (*best-result-criteria* %1)
                  (*best-result-criteria* %2))
        results))

(defn compile-results
  "Return an easier to use map of result data given
  from [[cross-validate-tests]].  The map returns all the performance
  statistics and:

  * **:feature-metas** feature metadatas
  * **:result** `weka.core.Evaluation` instance
  * **all-results** a sorted list of `weka.core.Evaluation` instances

  See [[cross-validate-tests]] for where the results data is created."
  [results]
  (map (fn [res]
         (merge
          ;; we can't add :eval since currently Evaluation isn't serializable
          (select-keys res [:accuracy :wprecision :wrecall :wfmeasure
                            :kappa :rmse :classifier :classify-attrib
                            :instances-total :instances-correct
                            :instances-incorrct :instances-trained
                            :train-total :test-total])
          {:feature-metas (map keyword (-> res :feature-metas))
           :result res
           :all-results results}))
       (reverse (sort-results results))))

(defn classifier-name
  "Return a decent human readable name of a classifier instance."
  [classifier-instance]
  (if (string? classifier-instance)
    classifier-instance
    (second (re-matches #".*\.(.+)" (.getName (.getClass classifier-instance))))))

(defn excel-results
  "Save the results in Excel format."
  [sheet-name-results out-file]
  (letfn [(create-sheet-data [results]
            (map (fn [result]
                   (letfn [(fp [key]
                             (format (str "%." excel-results-precision "f%%")
                                     (get result key)))
                           (f [key]
                             (format (str "%." excel-results-precision "f")
                                     (get result key)))]
                     (vec (apply concat
                                 `(~(fp :accuracy)
                                   ~(f :wprecision)
                                   ~(f :wrecall)
                                   ~(f :wfmeasure)
                                   ~(f :kappa)
                                   ~(f :rmse)
                                   ~(classifier-name
                                     (or (:classifier result)
                                         "<error:no classifier>"))
                                   ~(str/join ", "(:feature-metas result)))
                                 (list (map :value (:extra-cols result)))))))
                 (reverse (sort-results results))))
          (prepend-header [sheet-data headers]
            (vec (concat [(vec (map (fn [val]
                                      {:value (name val)
                                       :alignment :center
                                       :font {:bold true}})
                                    headers))]
                         sheet-data)))]
    (let [headers '(Accuracy Precision Recall F-Mesaure Kappa
                             RMSE Classifier Attributes)]
      (-> (excel/build-workbook
           (excel/workbook-hssf)
           (apply
            merge
            (map (fn [res]
                   (let [sheet-no-header (create-sheet-data (:results res))
                         extra-headers (map :header (:extra-cols (first (:results res))))
                         sheet-data (prepend-header sheet-no-header
                                                    (concat headers extra-headers))]
                     {(:sheet-name res) sheet-data}))
                 sheet-name-results)))
          (excel/save out-file))
      (log/infof "wrote results file: %s" out-file))))

(defn- print-eval-results
  "Print the results, confusion matrix and class details to standard out of a
  `weka.core.Evalution`."
  [eval]
  (println (.toSummaryString eval "\nResults\n" true))
  (println (.toMatrixString eval "Confusion Matrix"))
  (println (.toClassDetailsString eval "Class Details")))

(defn print-results
  "Print the results, confusion matrix and class details to standard out of a
  single or sequence of `weka.core.Evalution`s."
  [results & {:keys [title]}]
  (println (apply str (repeat 70 \=)))
  (when title
    (println title)
    (println (apply str (repeat 70 \=))))
  (let [res (if (sequential? results) results (list results))]
    (doseq [result res]
      (println (apply str (repeat 70 \-)))
      (print-eval-results (:eval result))
      ;;(println (format "recall: %s" (.recall (:eval result) 0)))
      (println (format "classifier: %s" (.getName (.getClass (:classifier result)))))
      (println (format "attributes: %s" (str/join ", "(:feature-metas result))))
      (doseq [keyval (dissoc result :classifier :feature-metas :eval :data)]
        (println (format "%s: %s" (name (first keyval)) (second keyval)))))))

(initialize)
