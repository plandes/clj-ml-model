(ns ^{:doc
"A *client* entry point library to help with executing a trained classifier.
The classifier is tested and trained in the [[zensols.model.eval-classifier]]
namespace.

This library expects that you configure your model.  To learn how to do that,
see [[with-model-conf]] and see the [repo
docs](https://github.com/plandes/clj-ml-model)."
      :author "Paul Landes"}
    zensols.model.execute-classifier
  (:use [clojure.java.io :as io :only [input-stream output-stream file]])
  (:use [clojure.pprint :only [pprint]])
  (:require [clojure.tools.logging :as log]
            [clojure.string :as str])
  (:require [zensols.actioncli.resource :refer (resource-path)]
            [zensols.model.classifier :as cl]
            [zensols.model.weka :as weka]))

(def ^{:dynamic true :private true} *model-config* nil)

(defmacro with-model-conf
  "Evaluates body with a model configuration.
  The model configuration is a map with the following keys:

  * **:name** human readable short name, which is used in file names and
  spreadsheet cells

  * **:create-feature-sets-fn** function creates a sequence of maps with each
  map having key/value pairs of the features of the model to be populated; it
  is passed with optional keys:
      * **:set-type** the data set type: `:test` or `:train`, which uses the [data set library](https://plandes.github.io/clj-ml-dataset/codox/zensols.dataset.db.html#var-instances)
      convention for easy integration

  * **:create-features-fn** just like **create-feature-sets-fn** but creates a
  single feature map used for test/execution after the classifier is built;
  it's called with the arguments that [[classify]] is given to classify ann
  instance along with the context generated at train time by **:context-fn** if
  it was provided (see below)

  * **:feature-metas-fn** a function that creates a map of key/value
  pairs describing the features where the values are `string`, `boolean`,
  `numeric`, or a sequence of strings representing possible enumeration
  values

  * **:display-feature-metas-fn** like **:feature-metas-fn** but used to
  display (i.e. while debugging)

  * **:class-feature-meta-fn** just like a **feature-metas-fn** but
  describes the class

  * **:context-fn** a function that creates a context (ie. stats on the entire
  training set) and passed to **:create-features-fn**

  * **:set-context-fn** (optional) a function that is called to set the context
  created with **:context-fn** and retrieved from the persisted model; this is
  useful when using/executing the model and the context is needed before
  **:create-features-fn** is called

  * **:model-return-keys** what the classifier will return (by default
  `{:label :distributions}`)

  * **:cross-fold-instances-inst** at atom used to cache the
  `weka.core.Instances` generated from **:create-feature-sets-fn**; when this
  atom is derefed as `nil` **:create-feature-sets-fn** is called to create the
  feature maps

  * **:feature-sets-set** a map of key/value pairs where keys are names of
  feature sets and the values are lists of lists of features as symbols"
  {:style/indent 1}
  [model-config & body]
  `(binding [*model-config* ~model-config]
     ~@body))

(defn model-config
  "Return the currently bound model configuration."
  []
  (if-not *model-config*
    (throw (ex-info "Model configuration not bound" {}))
    *model-config*))

(defn model-classifier-label
  "Return the class label metadata from the model config."
  []
  ((:class-feature-meta-fn (model-config))))

(defn model-classifier-feature-types
  "Return the feature metadatas from the model config."
  []
  (let [model-conf (model-config)]
    (->> ((:feature-metas-fn model-conf))
         (into {}))))

(defn- create-instances
  "Create the weka Instances object using sets of features created from
  **:create-feature-sets-fn**.  If **context** is given call :set-context-fn.

  See [[with-model-conf]], which explains the keys."
  ([features-set]
   (create-instances features-set nil))
  ([features-set context]
   (log/infof "generated instances from %d feature sets" (count features-set))
   (log/tracef "feature sets: <<%s>>" (pr-str features-set))
   (let [{:keys [name set-context-fn]} (model-config)]
     (if (and set-context-fn context)
       (set-context-fn context))
     (weka/instances
      (format "%s-classify" name)
      features-set
      (model-classifier-feature-types)
      (model-classifier-label)))))

(defn cross-fold-instances
  "Called by [[eval-classifier]] to create the data set for cross validation.
  See [[create-instances]]."
  []
  (log/info "generating feature sets from model config")
  (let [{:keys [cross-fold-instances-inst create-feature-sets-fn]} (model-config)]
    (assert cross-fold-instances-inst
            "No :cross-fold-instances-inst atom set on model configuration")
    (swap! cross-fold-instances-inst
           #(or % (create-instances (create-feature-sets-fn))))))

(defn train-test-instances
  "Called by [[eval-classifier]] to create the data set for cross validation.
  See [[create-instances]]."
  []
  (log/info "generating feature sets from model config")
  (let [{:keys [test-train-instances-inst create-feature-sets-fn]} (model-config)]
    (assert test-train-instances-inst
            "No :instances-inst atom set on model configuration")
    (swap! test-train-instances-inst
           #(or % {:train (create-instances (create-feature-sets-fn :set-type :train))
                   :test (create-instances (create-feature-sets-fn :set-type :test))}))))

(defn model-exists?
  "Return whether a model file exists on the file system."
  []
  (cl/model-exists? (:name (model-config))))

(defn read-model
  "Read/unpersist the model from the file system."
  [& {:keys [fail-if-exist?] :or {fail-if-exist? false}}]
  (cl/read-model (:name (model-config)) :fail-if-exist? fail-if-exist?))

(defn prime-model
  "Prime a trained or unpersisted ([[read-model]]) model for classification
  with [[classify]]."
  [model]
  (binding [weka/*missing-values-ok* true]
    (let [{classifier :classifier
           feature-metas :feature-metas
           classify-attrib :classify-attrib
           context :context} model
          model-conf (model-config)
          attrib-keys (keys (keyword feature-metas))
          ;; creates an Instances with a single row/Instance of nulls
          feature-set (zipmap attrib-keys (repeat (count attrib-keys) nil))
          features-set (list feature-set)
          attribs (map name feature-metas)
          unfiltered (create-instances features-set context)]
      (binding [cl/*class-feature-meta* (name classify-attrib)]
        (let [instances (cl/filter-attribute-data unfiltered attribs)]
          (merge model
                 {:model-conf (model-config)
                  :instances instances}))))))

(defn print-model-info
  "Print informtation from a (usually serialized) model.  This data includes
  performance metrics, the classifier, features used to create the model and
  the context (see [[zensols.model.execute-classifier]])."
  [model]
  (doseq [key [:instances-total :instances-correct :instances-incorrct
               :name :create-time :accuracy :wprecision :wrecall :wfmeasure]]
    (println (format "%s: %s" (name key) (get model key))))
  (println "features:")
  (println (:feature-metas model))
  (println "classifier:")
  (println (:classifier model))
  (println "context:")
  (println (:context model)))

(defn dump-model-info
  "Write all data from [[print-model-info]] to the file system.

  See [[zensols.model.classifier/modeldir]] for where the model is read from
  and [[zensols.model.classifier/analysis-report-resource]] for information about to
  where the model information is written."
  [model]
  (let [model-conf (:model-conf model)
        outfile (io/file (cl/analysis-report-resource)
                         (format "%s-model.txt" (:name model-conf)))]
    (with-open [writer (io/writer outfile)]
      (binding [*out* writer]
        (print-model-info model))
      (.flush writer))
    (log/infof "wrote model dump to file %s" outfile)
    outfile))

(defn write-classifier
  "Serialize the model to the file system.

  * **model** a model created from [[zensols.model.eval-classifier/train-model]"
  [model]
  (let [file (io/file (cl/analysis-report-resource)
                      (format "%s-classifier.dat" (:name model)))]
   (with-open [out (output-stream file)]
     (let [out-obj (java.io.ObjectOutputStream. out)]
       (.writeObject out-obj (:classifier model))))))

(defn- set-instance-values [instance features]
  (log/tracef "features: %s" features)
  (doseq [attrib (map #(.attribute instance %)
                      (range (.numAttributes instance)))]
    (let [feature-key (keyword (.name attrib))
          feature (get features feature-key)
          val (weka/value-for-instance feature)]
      (log/tracef "setting feature %s (%s): %s"
                  (.name attrib) feature-key feature)
      (when (not (nil? feature))
        (log/debugf "feature-meta %s: setting <%s> (%s) -> %s"
                    (.name attrib) feature (type feature) val)
        (try (.setValue instance attrib val)
             (catch Exception e
               (let [msg (format "Can't set value <%s> for attrib <%s>: %s"
                                 val attrib (.toString e))]
                 (log/error e msg)
                (throw (ex-info msg {:val val
                                     :attrib attrib}
                                e)))))))))

(defn classify
  "Classify a single instance using a trained model.

  * **model** a model created
  from [[zensols.model.eval-classifier/train-model]] or [[read-model]]"
  [model & data]
  (log/debugf "classifying: %s" data)
  (with-model-conf (:model-conf model)
    (let [{classifier :classifier
           feature-metas :feature-metas
           ;; this was the context that was persisted with the model
           context :context} model
          _ (log/tracef "context: <%s>" context)
          model-conf (model-config)
          trans-fn (or (:classifications-map-fn model-conf) identity)
          create-features-fn (or (:create-features-fn model-conf)
                                 (throw (ex-info "No create-features-fn defined in model"
                                                 {:model-name (:name model-conf)})))
          _ (log/debugf "create-features-fn: %s" create-features-fn)
          cfargs (concat data (list context))
          _ (log/debugf "arg count: %d" (count cfargs))
          features (apply create-features-fn cfargs)
          ;; must use copy constructor or it gives bad results
          instances (weka/clone-instances (:instances model))
          attribs (map name feature-metas)
          return-keys (:model-return-keys model-conf)]
      (set-instance-values (.instance instances 0) features)
      (->> (cl/classify-instance classifier instances return-keys)
           first
           (merge (if (contains? return-keys :features) {:features features}))
           trans-fn ))))
