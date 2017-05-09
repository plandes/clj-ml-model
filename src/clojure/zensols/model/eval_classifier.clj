(ns ^{:doc
      "A *client* entry point library to help with evaluating machine learning
models.  This library not only wraps the Weka library but also provides
additional functionality like a two pass cross
validation (see [[with-two-pass]])."
      :author "Paul Landes"}
    zensols.model.eval-classifier
  (:require [clojure.tools.logging :as log]
            [clojure.java.io :as io :refer [input-stream output-stream file]]
            [clojure.string :as s]
            [clojure.stacktrace :refer (print-stack-trace)]
            [clojure.data.csv :as csv])
  (:require [zensols.tabres.display-results :as dr])
  (:require [zensols.model.execute-classifier :refer (model-config) :as exc]
            [zensols.model.weka :as weka]
            [zensols.model.classifier :as cl]))

(def ^:dynamic *default-set-type*
  "The default type of test, which is one of:

      * `:cross-validation`: run a N fold cross validation (default)
      * `:train-test`: train the classifier and then evaluate"
  :cross-validation)

(def ^:dynamic *throw-cross-validate*
  "If `true`, throw an exception during cross validation for any errors.
  Otherwise, the error is logged and cross-validation continues.  This is
  useful for when classifiers are used and some choke given the dataset, but
  you still want the other results."
  false)

(defn- by-set-type-instances [set-type]
  (case set-type
    :cross-validation (exc/cross-fold-instances)
    :train-test (:train-test (exc/train-test-instances))
    :test (:test (exc/train-test-instances))
    :train (:train (exc/train-test-instances))
    true (throw (ex-info "Unknon set type"
                         {:set-type *default-set-type*}))))

(defn print-model-config
  "Pretty print the model configuation set
  with [[zensols.model.execute-classifier/with-model-conf]]."
  []
  (clojure.pprint/pprint (model-config)))

(defn analysis-file
  ([]
   (analysis-file "%s-data.arff"))
  ([file-format]
   (let [model-conf (model-config)
         file-name (format file-format (:name model-conf))]
     (io/file (cl/analysis-report-resource) file-name))))

(defn read-arff
  "Read the ARFF file configured
  with [[zensols.model.execute-classifier/with-model-conf]].  If **file** is
  given, use that file instead of getting it
  from [[zensols.model.classifier/analysis-report-resource]]."
  ([]
   (read-arff (analysis-file)))
  ([file]
   (cl/read-arff file)))

(defn write-arff
  "Write the ARFF file configured
  with [[zensols.model.execute-classifier/with-model-conf]].  If **file** is
  given, use that file instead of getting it
  from [[zensols.model.classifier/analysis-report-resource]]."
  ([]
   (write-arff (analysis-file)))
  ([file]
   (binding [cl/*arff-file* file]
     (cl/write-arff (by-set-type-instances :train-test)))
   file))

(defn- feature-matrix
  "Generate a matrix of features as configured in a model with
  [[zensols.model.execute-classifier/with-model-conf]]."
  [& {:keys [max] :as adb-keys
      :or {max Integer/MAX_VALUE}}]
  (let [{:keys [feature-metas-fn display-feature-metas-fn
                class-feature-meta-fn create-feature-sets-fn]} (model-config)
        display-feature-metas-fn (or display-feature-metas-fn feature-metas-fn)
        feature-metas (display-feature-metas-fn)
        class-feature-meta (class-feature-meta-fn)
        adb-keys (if adb-keys
                   (->> (dissoc adb-keys :max)
                        (into [])))
        feature-sets (->> adb-keys
                          (apply create-feature-sets-fn)
                          (take max))]
    (let [keys (map first (concat (list class-feature-meta) feature-metas))]
      (->> feature-sets
           (map (fn [tok]
                  (map (fn [tkey]
                         (get tok tkey))
                       keys)))
           (hash-map :column-names (map name keys) :data)))))

(defn display-features
  "Display features as configured in a model with
  [[zensols.model.execute-classifier/with-model-conf]].

  **adb-keys** are given to `:create-feature-sets-fn` as described
  in [[zensols.model.execute-classifier/with-model-conf]].  In addition it
  includes `:max`, which is the maximum number of instances to display."
  [& adb-keys]
  (let [{:keys [column-names data]} (apply feature-matrix adb-keys)]
    (dr/display-results data :column-names column-names)))

(defn features-file
  "Return the default file used to create the features output file
  with [[write-features]]."
  []
  (analysis-file "%s-features.csv"))

(defn write-features
  "Write features as configured in a model with [[zensols.model.execute-classifier/with-model-conf]] to a CSV
  spreadsheet file.

  See [[features-file]] for the default file

  For the non-zero-arg form, see [[zensols.model.execute-classifier/with-model-conf]]."
  ([]
   (write-features (features-file)))
  ([file]
   (let [{:keys [column-names data]} (feature-matrix)]
     (with-open [writer (io/writer file)]
       (->> data
            (cons column-names)
            (csv/write-csv writer)))
     (log/infof "wrote features file: %s" file)
     file)))

(defn- cross-validate-results-struct
  "Create a data structure with cross validation results."
  [classifiers attribs-sets feature-metadata]
  (log/debugf "cvr struct classifiers=<%s>, attribs=%s"
              (pr-str classifiers) (pr-str attribs-sets))
  (letfn [(cr-test [classifier attribs]
            (try
              (log/infof "classifier: <%s>, %s" classifier (pr-str attribs))
              (let [res (cl/cross-validate-tests classifier (map name attribs)
                                                 feature-metadata)]
                (log/debug (with-out-str (cl/print-results res)))
                res)
              (catch Exception e
                (if *throw-cross-validate*
                  (throw e)
                  (let [msg (format "Can't cross validate classifier %s, attrib: %s: message=%s"
                                    classifier (if attribs (s/join "," attribs))
                                    (.toString e))]
                    (log/error (str "stack trace: " (print-stack-trace e)))
                    (log/error e msg)
                    nil)))))]
    (->> classifiers
         (map (fn [classifier]
                (keep #(cr-test classifier %) attribs-sets)))
         (remove empty?)
         (apply concat))))

(defn- freeze-feature-metadata [model-conf]
  (let [{:keys [feature-metas-fn class-feature-meta-fn]} model-conf
        class-feat (class-feature-meta-fn)]
   {:feature-metas (into {} (feature-metas-fn))
    :class-key (first class-feat)
    :class-type (second class-feat)}))

(declare two-pass-config)
(declare executing-two-pass?)

(defn- cross-validate-results
  "Cross validate several models in series.

  * **classifier-sets** is a key in [[zensols.model.weka/*classifier*]] or a
  constructed classifier (see [[zensols.model.weka/make-classifiers]])
  * **feature-sets-key** identifies what feature set (see
  **:feature-sets-set** in [[zensols.model.execute-classifier/with-model-conf]])"
  [classifier-sets feature-sets-key]
  (log/debugf "cvr struct attrib key=%s" feature-sets-key)
  (let [model-conf (model-config)]
    (log/debugf "model config keys: %s" (keys model-conf))
    (let [classifier-attrib (first (exc/model-classifier-label))
          feature-meta-sets (get (:feature-sets-set model-conf)
                                 feature-sets-key)
          feature-metadata (freeze-feature-metadata model-conf)
          _ (assert feature-meta-sets (format "no such feature meta set: <%s>"
                                              feature-sets-key))
          instances (exc/cross-fold-instances)
          num-inst (.numInstances instances)
          folds cl/*cross-fold-count*]
      (log/debugf "number of instances: %d, feature-meta-set: <%s>, attribs: %s"
                  num-inst (pr-str feature-meta-sets)
                  (->> (weka/attributes-for-instances instances)
                       (map :name)
                       pr-str))
      (log/tracef "instances class: %s" (-> instances .getClass .getName))
      (if-let [id-val (if (and (executing-two-pass?)
                               (> (.numInstances instances) 0))
                        (->> (two-pass-config)
                             :id-key
                             name
                             (weka/value instances 0)))]
        (if (= "?" id-val)
          (-> (format "Key not set yet for two pass cross validation: %s"
                      (.instance instances 0))
              (ex-info {:instance (.instance instances 0)})
              throw)))
      (if (<= num-inst folds)
        (throw (ex-info "Not enough folds"
                        {:num-inst num-inst
                         :folds folds}))
        (binding [cl/*get-data-fn* #(identity instances)
                  cl/*class-feature-meta* (name classifier-attrib)]
          (->> classifier-sets
               (map #(cross-validate-results-struct
                      (weka/make-classifiers %)
                      feature-meta-sets
                      feature-metadata))
               (apply concat)
               doall))))))

(defn train-test-results
  "Test the performance of a model by training on a given set of data
  and evaluate on the test data.

  See [[train-model]] for parameter details."
  [classifier-sets feature-sets-key]
  (log/debugf "feature-sets-key=%s, classifier-sets=%s"
              feature-sets-key classifier-sets)
  (let [model-conf (model-config)
        _ (log/debugf "model config keys: %s" (keys model-conf))
        classifier-attrib (first (exc/model-classifier-label))
        feature-meta-sets (get (:feature-sets-set model-conf)
                               feature-sets-key)
        feature-metadata (freeze-feature-metadata model-conf)
        _ (log/debugf "feature meta sets: <%s>"
                      (pr-str feature-meta-sets))
        {train-instances :train test-instances :test}
        (exc/train-test-instances)]
    (assert feature-meta-sets (format "no such feature meta set: <%s>"
                                      feature-sets-key))
    (assert train-instances "train-instances")
    (assert test-instances "test-instances")
    (log/infof "number of train/test instances:(%d, %d)"
               (.numInstances train-instances)
               (.numInstances test-instances))
    (log/debugf "feature metas: %s " (pr-str feature-meta-sets))
    (log/tracef "train instances class: %s"
                (-> train-instances .getClass .getName))
    (log/tracef "test instances class: %s"
                (-> test-instances .getClass .getName))
    (binding [cl/*class-feature-meta* (name classifier-attrib)]
      (->> classifier-sets
           (map weka/make-classifiers)
           (apply concat)
           (map #(cl/train-test-classifier % feature-meta-sets
                                           feature-metadata
                                           train-instances test-instances))
           (apply concat)
           doall))))

(defn run-tests
  "Create result sets useful to functions like [[eval-and-write]].  This
  package was designed for most use cases to not have to use this function.

  See [[*throw-cross-validate*]]."
  [classifier-sets feature-set-key]
  (let [test-fn (case *default-set-type*
                  :cross-validation cross-validate-results
                  :train-test train-test-results)
        res (test-fn classifier-sets feature-set-key)]
    (log/debugf "results count %d" (count res))
    res))

(defn compile-results
  "Run cross-fold validation and compile into a nice results map sorted by
  performance.

  See [[zensols.model.classifier/compile-results]].

  * **classifier-sets** is a key in [[zensols.model.weka/*classifiers*]] or a
  constructed classifier (see [[zensols.model.weka/make-classifiers]])

  * **feature-sets-key** identifies what feature set (see
  **:feature-sets-set** in [[zensols.model.execute-classifier/with-model-conf]])"
  [classifier-sets feature-set-key]
  (->> (run-tests classifier-sets feature-set-key)
       cl/compile-results))

(defn terse-results
  "Return terse cross-validation results in an array:
  * classifier name
  * weighted F-measure
  * feature-metas

  * **classifier-sets** is a key in [[zensols.model.weka/*classifiers*]] or a
  constructed classifier (see [[zensols.model.weka/make-classifiers]])

  * **feature-sets-key** identifies what feature set (see
  **:feature-sets-set** in [[zensols.model.execute-classifier/with-model-conf]])

  ## Keys

  * **:only-stats?** if `true` only return statistic data

  See [[*throw-cross-validate*]]."
  [classifier-sets feature-set-key &
   {:keys [only-stats?]
    :or {only-stats? true}}]
  (let [res (compile-results classifier-sets feature-set-key)]
    (log/debugf "terse results count %d" (count res))
    (map (fn [elt]
           (concat [(cl/classifier-name (:classifier elt))
                    (:wfmeasure elt)]
                   (if-not only-stats?
                     [feature-set-key
                      (:attributes elt)])))
         res)))

(defn print-best-results
  "Print the highest (best) scored cross validation information.

  * **classifier-sets** is a key in [[zensols.model.weka/*classifiers*]] or a
  constructed classifier (see [[zensols.model.weka/make-classifiers]])
  * **feature-sets-key** identifies what feature set (see
  **:feature-sets-set** in [[zensols.model.execute-classifier/with-model-conf]])

  See [[*throw-cross-validate*]]."
  [classifier-sets feature-set-key]
  (let [comp-res (compile-results classifier-sets feature-set-key)]
    (cl/print-results (:result (first comp-res)))))

(defn create-model
  "Create a model that can be trained.  This runs cross fold validations to
  find the best classifier and feature set into a result that can be used
  with [[train-model]] and subsequently [[write-model]].

  * **classifier-sets** is a key in [[zensols.model.weka/*classifiers*]] or a
  constructed classifier (see [[zensols.model.weka/make-classifiers]])
  * **feature-sets-key** identifies what feature set (see
  **:feature-sets-set** in [[zensols.model.execute-classifier/with-model-conf]])

  See [[*throw-cross-validate*]]."
  [classifier-sets feature-set-key]
  (-> (compile-results classifier-sets feature-set-key)
      first
      (dissoc :result :all-results)
      (merge {:name (:name (model-config))
              :create-time (java.util.Date.)})))

(defn train-model
  "Train a model created from [[create-model]].  The model is trained on the
  full available dataset.  After the classifier is trained, you can save it to
  disk by calling [[write-model]].

  * **model** a model that was created with [[create-model]]

  See [[*throw-cross-validate*]]."
  [model & {:keys [set-type] :or {set-type *default-set-type*}}]
  (let [classifier (:classifier model)
        attribs (map name (:attributes model))
        classify-attrib (first (exc/model-classifier-label))
        instances (by-set-type-instances set-type)]
    (binding [cl/*get-data-fn* #(identity instances)
              cl/*class-feature-meta* (name classify-attrib)]
      (log/infof "training model %s classifier %s with %d instances"
                 (:name (model-config))
                 (-> classifier .getClass .getName)
                 (.numInstances instances))
      (cl/train-classifier classifier attribs)
      model)))

(defn- model-persist-name
  "Return the name of the model of the model configuration."
  []
  (:name (model-config)))

(defn write-model
  "Persist/write the model to disk.

  * **model** a model that was trained with [[train-model]]

  See [[zensols.model.classifier/model-dir]] for information about to where the
  model is written."
  ([model]
   (write-model model (model-persist-name)))
  ([model name]
   (let [model-conf (model-config)
         context-fn (:context-fn model-conf)]
     (log/tracef "saving classifer %s as %s"
                 (type (:classifier model)) name)
     (let [context (if context-fn (context-fn))
           model (if context
                   (assoc model :context context)
                   model)]
       (cl/write-model name model))
     model)))

(defn read-model
  "Read a model that was previously persisted to the file system.

  See [[zensols.model.classifier/model-dir]] for where the model is read from."
  []
  (cl/read-model (model-persist-name)))

(defn evaluations-file
  "Return the default file used to create an evaluations file
  with [[eval-and-write]]."
  ([]
   (evaluations-file "classification"))
  ([fname]
   (let [model-conf (model-config)]
     (io/file (cl/analysis-report-resource)
              (format "%s-%s.xls" (:name model-conf) fname)))))

(defn eval-and-write-results
  "Perform a cross validation and write the results to an Excel formatted file.
  The data from **results** is obtained with [[run-tests]].

  See [[eval-and-write]] and [[*throw-cross-validate*]]."
  ([results]
   (eval-and-write-results results (evaluations-file)))
  ([results output-file]
   (let [model-conf (model-config)]
     (cl/excel-results
      [{:sheet-name (format "%s Classification" (s/capitalize (:name model-conf)))
        :results results}]
      output-file)
     (log/infof "wrote results file: %s" output-file)
     output-file)))

(defn eval-and-write
  "Perform a cross validation and write the results to an Excel formatted file.

  See [[zensols.model.classifier/analysis-report-resource]] for where the file is
  written.

  * **classifier-sets** is a key in [[zensols.model.weka/*classifiers*]] or a
  constructed classifier (see [[zensols.model.weka/make-classifiers]])
  * **feature-sets-key** identifies what feature set (see
  **:feature-sets-set** in [[zensols.model.execute-classifier/with-model-conf]])

  This uses [[eval-and-write-results]] to actually write the results.

  See [[evaluations-file]] and [[*throw-cross-validate*]]."
  ([classifier-sets set-key]
   (eval-and-write-results (run-tests classifier-sets set-key)))
  ([classifier-sets set-key file]
   (eval-and-write-results (run-tests classifier-sets set-key) file)))

(defn train-test-series
  "Test and train with different rations and return the results.  The return
  data writable directly as an Excel file.  However, you can also save it as a
  CSV with [[write-csv-train-test-series]].

  The keys are the classifier name and the values are the 2D result matrix.

  See [[*throw-cross-validate*]]."
  [classifiers meta-set divide-ratio-config]
  (let [{:keys [start stop step]} divide-ratio-config
        {:keys [divide-by-set]} (model-config)
        stat-keys [:instances-trained :instances-tested
                   :wfmeasure :wprecision :wrecall]
        stat-header ["Train" "Test" "F-Measure" "Precision" "Recall"]]
    (->> classifiers
         (map (fn [classifier]
                (->> (range (* start 100) (* stop 100) (* step 100))
                     (map #(/ % 100))
                     (map (fn [divide-ratio]
                            (log/infof "dividing train/test split by %.3f"
                                       divide-ratio)
                            (divide-by-set divide-ratio)
                            (binding [*default-set-type* :train-test]
                              (compile-results (list classifier) meta-set))))
                     (apply concat)
                     (map (fn [res]
                            (let [name (-> res :classifier cl/classifier-name)]
                              {:name name
                               :stats (map #(get res %) stat-keys)})))
                     (#(hash-map (-> % first :name)
                                 (->> (map :stats %)
                                      (cons stat-header)))))))
         (apply merge))))

(defn test-train-series-file
  "Return the default file used to create an evaluations file
  with [[eval-and-write]]."
  ([]
   (test-train-series-file "train-test-series"))
  ([fname]
   (let [model-conf (model-config)]
     (io/file (cl/analysis-report-resource)
              (format "%s-%s.csv"
                      (:name model-conf) fname)))))

(defn write-csv-train-test-series
  "Write the results produced with [[train-test-series]] as a CSV file to the
  analysis directory."
  ([res]
   (write-csv-train-test-series res (test-train-series-file)))
  ([res out-file]
   (->> res
        (map (fn [[classifier-name data]]
               (with-open [writer (io/writer out-file)]
                 (csv/write-csv writer data))))
        doall)))



;; two pass

(defn two-pass-model
  "Don't use this function--instead, use [[with-two-pass]].

  Create a two pass model, which should be merged with the model created
  with [[zensols.model.execute-classifier/with-model-conf]].

  See [[with-two-pass]]."
  [model id-key anon-by-id-fn anons-fn]
  (let [id-key (-> id-key name symbol)]
    (->> model
         :feature-sets-set
         (map (fn [[k v]]
                {k (map #(cons id-key %) v)}))
         (apply merge)
         (hash-map :two-pass-config
                   {:anon-by-id-fn anon-by-id-fn
                    :anons-fn anons-fn
                    :id-key id-key}
                   :feature-sets-set)
         (merge model))))

(defn- two-pass-config
  "Assert and return the two pass configuration."
  [& {:keys [nothrow?]}]
  (let [conf (:two-pass-config (model-config))]
    (if conf
      conf
      (if-not nothrow?
        (-> "No two pass model found--use `two-pass-model`"
            (ex-info {:model-config (model-config)})
            throw)))))

(defn executing-two-pass?
  "Return `true` if we're currently using a two pass cross validation."
  []
  (let [tpconf (two-pass-config :nothrow? true)]
    (not (nil? tpconf))))

(defn- anon-ids-for-instance
  "Return IDs as integers found in a `core.weka.Instances`."
  [insts id-attrib]
  (->> (range (.numInstances insts))
       (map (fn [row]
              (-> insts
                  (.instance row)
                  (.stringValue id-attrib))))))

(defn two-pass-train-instances
  "Don't use this function--instead, use [[with-two-pass]].

  This is called by the [[zensols.model.weka]] namespace."
  [insts state org folds fold]
  (log/infof "training set: %d, org=%d"
              (.numInstances insts)
              (.numInstances org))
  (let [{:keys [create-two-pass-context-fn create-feature-sets-fn]} (model-config)
        _ (assert create-feature-sets-fn)
        tpconf (two-pass-config)
        _ (assert tpconf)
        {:keys [id-key anons-fn]} tpconf
        _ (assert id-key)
        _ (assert anons-fn)
        id-key-att-name (name id-key)
        id-attrib (weka/attribute-by-name insts id-key-att-name)
        _ (assert id-attrib (format "weka attribute for <%s>" id-key-att-name))
        anon-ids (anon-ids-for-instance insts id-attrib)
        _ (log/infof "training %d instances for fold %d of %d"
                     (count anon-ids) (inc fold) folds)
        _ (log/debugf "anon-ids: %d instances" (count anon-ids))
        _ (log/debugf "maybe invoking create context with: %s"
                      create-two-pass-context-fn)
        _ (log/tracef "ids: <%s>" (s/join ", " anon-ids))
        context (if create-two-pass-context-fn
                  (create-two-pass-context-fn :anons-fn anons-fn
                                              :id-set anon-ids))
        feature-metas (if context
                        (exc/model-classifier-feature-types context)
                        exc/model-classifier-feature-types)
        data-maps (create-feature-sets-fn :ids anon-ids :context context)]
    (swap! state assoc
           :context context
           :feature-metas feature-metas)
    (-> insts
        (weka/remove-attributes [id-key-att-name])
        (weka/populate-instances feature-metas data-maps))))

(defn two-pass-test-instances
  "Don't use this function--instead, use [[with-two-pass]].

  This is called by the [[zensols.model.weka]] namespace."
  [insts train-state org folds fold]
  (log/debugf "testing set: %d" (.numInstances insts))
  (let [{:keys [context feature-metas]} @train-state
        {:keys [create-features-fn]} (model-config)
        {:keys [id-key anons-fn anon-by-id-fn]} (two-pass-config)
        id-key-att-name (name id-key)
        id-attrib (weka/attribute-by-name insts id-key-att-name)
        anon-ids (anon-ids-for-instance insts id-attrib)
        _ (log/infof "testing %d instances for fold %d of %d"
                     (count anon-ids) (inc fold) folds)
        _ (log/debugf "testing instances for ids: <%s>" (s/join ", " anon-ids))
        data-maps (->> anon-ids
                       (map (fn [id]
                              (log/debugf "creating features for id: %s" id)
                              (let [anon (anon-by-id-fn id)]
                                (or anon
                                    (-> (format "No annotation in DB: %d (%s)"
                                                id (type id))
                                        (ex-info {:id id})
                                        throw)))))
                       (map #(create-features-fn % context)))]
    (-> insts
        (weka/remove-attributes [id-key-att-name])
        (weka/populate-instances feature-metas data-maps))))

(defmacro with-two-pass
  "Like `with-model-conf`, but compute a context state (i.e. statistics needed
  by the model) on a per fold when executing a cross fold validation.

  The `model-conf` parameter is the same model used
  with [[zensols.model.execute-classifier/with-model-conf]].


  ## Description

  Two pass validation is a term used in this library.  During cross-validation
  the entire data set is evaluated and (usually) statistics or some other
  additional modeling happens.

  Take for example you want to count words (think Naive Bays spam filter).  If
  create features for the entire dataset before cross-validation you're
  *cheating* because the features are based on data not seen from the test
  folds.

  To get more accurate performance metrics you can provide functions that takes
  the current training fold, compute your word counts and create your features.
  During the testing phase, the computed data is provided to create features
  based on only that (current) fold.

  To use two pass validation ever feature set needs a unique key (not needed as
  a feature).  This key is then given to a function during validation to get
  the corresponding feature set that is to be *stitched* in.

  **Note** This is *only* useful if:

  1. You want to use cross fold validation to test your model.
  2. Your model priors (*context* in implementation parlance) is composed of
     the dataset preproessing, and thus, needed to get reliable performance
     metrics.


  ## Option Keys

  In addition to all keys documented
  in [[zensols.model.execute-classifier/with-model-conf]], the **opts** param
  is a map that also needs the following key/value pairs:

  * **:id-key** a function that takes a key as input and returns a feature set
  * **:anon-by-id-fn** is a function that takes a single integer argument of the
  annotation to retrieve by ID
  * **:anons-fn** is a function that retrieves all annotations
  * **:create-two-pass-context-fn** like `:create-context-fn`, as documented
  in [[zensols.model.execute-classifier/with-model-conf]] but called for two
  pass cross validation; this allows a more general context and a specific two
  pass context to be created for the unique needs of the model

  ## Example

  ```
  (with-two-pass (create-model-config)
    (if true
      {:id-key sf/id-key
       :anon-by-id-fn #(->> % adb/anon-by-id :instance)
       :anons-fn adb/anons})
  (with-feature-context (sf/create-context :anons-fn adb/anons
                                           :set-type :train-test)
    (ec/terse-results [:j48] :set-test-two-pass :only-stats? true)))
  ```

  See a [working example](https://github.com/plandes/clj-example-nlp-ml/blob/master/src/clojure/zensols/example/sa_tp_eval.clj)
  for a more comprehensive code listing."
  {:style/indent 2}
  [model-conf opts & forms]
  `(let [opts# (eval ~opts)]
     (if-not (eval opts#)
       (exc/with-model-conf ~model-conf
         ~@forms)
       (let [id-key# (:id-key opts#)
             anon-by-id-fn# (:anon-by-id-fn opts#)
             anons-fn# (:anons-fn opts#)
             model-conf# (two-pass-model ~model-conf id-key#
                                         anon-by-id-fn# anons-fn#)]
         (binding [weka/*missing-values-ok* true
                   cl/*cross-val-fns* {:train-fn two-pass-train-instances
                                       :test-fn two-pass-test-instances}]
           (exc/with-model-conf model-conf#
             ~@forms))))))
