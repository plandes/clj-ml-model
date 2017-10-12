(defproject com.zensols.ml/model "0.1.0-SNAPSHOT"
  :description "Interface for machine learning modeling, testing and training."
  :url "https://github.com/plandes/clj-ml-model"
  :license {:name "Apache License version 2.0"
            :url "https://www.apache.org/licenses/LICENSE-2.0"
            :distribution :repo}
  :plugins [[lein-codox "0.10.3"]
            [lein-javadoc "0.3.0"]
            [org.clojars.cvillecsteele/lein-git-version "1.2.7"]]
  :codox {:metadata {:doc/format :markdown}
          :project {:name "Interface for machine learning modeling, testing and training"}
          :output-path "target/doc/codox"
          :source-uri "https://github.com/plandes/clj-ml-model/blob/v{version}/{filepath}#L{line}"}
  :javadoc-opts {:package-names ["zensols.weka"]
                 :output-dir "target/doc/apidocs"}
  :git-version {:root-ns "zensols.model"
                :path "src/clojure/zensols/model"
                :version-cmd "git describe --match v*.* --abbrev=4 --dirty=-dirty"}
  :source-paths ["src/clojure"]
  :java-source-paths ["src/java"]
  :javac-options ["-Xlint:unchecked"]
  :exclusions [nz.ac.waikato.cms.weka/weka-dev]
  :dependencies [[org.clojure/clojure "1.8.0"]

                 ;; command line
                 [com.zensols.tools/actioncli "0.0.19"]

                 ;; dev
                 [com.zensols.gui/tabres "0.0.6"]

                 ;;; reports
                 [outpace/clj-excel "0.0.2"]
                 [org.clojure/data.csv "0.1.2"]
                 [com.zensols.tools/misc "0.0.5"]

                 ;; model serialization
                 [com.taoensso/nippy "2.13.0"]

                 ;; ML
                 [nz.ac.waikato.cms.weka/weka-stable "3.8.1"
                  :exclusions [com.github.fommil.netlib/core]]

                 ;; weka classifiers are separate deps starting with 3.7
                 [nz.ac.waikato.cms.weka/ridor "1.0.2"]
                 [nz.ac.waikato.cms.weka/hyperPipes "1.0.1"]
                 [nz.ac.waikato.cms.weka/conjunctiveRule "1.0.2"]
                 [nz.ac.waikato.cms.weka/NNge "1.0.2"]
                 [nz.ac.waikato.cms.weka/grading "1.0.2"]
                 [nz.ac.waikato.cms.weka/simpleEducationalLearningSchemes "1.0.1"]
                 [nz.ac.waikato.cms.weka/DTNB "1.0.2"]

                 [nz.ac.waikato.cms.weka/LibSVM "1.0.10"
                  :exclusions [tw.edu.ntu.csie/libsvm]]
                 [tw.edu.ntu.csie/libsvm "3.17"]]
  :profiles {:appassem {:aot :all}
             :snapshot {:git-version {:version-cmd "echo -snapshot"}}
             :dev
             {:jvm-opts
              ["-Dlog4j.configurationFile=test-resources/log4j2.xml" "-Xms4g" "-Xmx12g" "-XX:+UseConcMarkSweepGC"]
              :dependencies [[nz.ac.waikato.cms.weka/weka-stable "3.8.1" :classifier "sources"]
                             [org.apache.logging.log4j/log4j-core "2.7"]
                             [org.apache.logging.log4j/log4j-slf4j-impl "2.7"]]}})
