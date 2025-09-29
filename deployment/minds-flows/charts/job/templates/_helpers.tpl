{{/*
Define the standardized name of this helm chart and its objects
*/}}
{{- define "name" -}}
{{- required "A valid Values.name is required!" .Values.name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Define the standardized namespace, this is NOT defined in the output yaml files,
but it is used inside some variables and eg: for the urls generated for our
ingresses (namespaced subdomain urls, etc)
*/}}
{{- define "namespace" -}}
{{- required "A valid global.namespace is required!" .Values.global.namespace -}}
{{- end -}}

{{/*
A helper to print env variables TODO MAKE THIS WORKS
Exmaple: {{- include "print_envs" .Values.globalEnvs | indent 12 }}
*/}}
{{- define "print_envs" -}}
{{- range . }}
- name: {{ .name | quote }}
  {{- if .value }}
  value: {{ with .value }}{{ tpl . $ | quote }}{{- end }}
  {{- end }}
  {{- if .valueFrom }}
  valueFrom:
{{ .valueFrom | toYaml | indent 16 }}
  {{- end }}
{{- end }}
{{- end -}}

{{/*
Create chart name and version as used by the chart label.  Version is optional because statefulsets don't like labels being updated dynamically
*/}}
{{- define "chart" -}}
{{ if and (hasKey .Values "labelsIncludeChartVersion") (eq (coalesce .Values.labelsIncludeChartVersion "1" | toString) "1") }}
{{- printf "%s" .Chart.Name | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{/*
Generate basic labels for pods/services/etc
Sample Usage: {{- include "labels" . | indent 2 }}
*/}}
{{- define "labels" }}
labels:

{{- if .Values.usingNewRecommendedLabels }}
# see: https://kubernetes.io/docs/concepts/overview/working-with-objects/common-labels/

{{- if .Values.labelsEnableDefault }}
  app.kubernetes.io/name: {{ .Values.name | trunc 63 | trimSuffix "-" | quote }}
  app.kubernetes.io/instance: {{ .Values.name | trunc 63 | trimSuffix "-" | quote }}
{{- end }}

{{- else }}

{{- if .Values.labelsEnableDefault }}
  app: {{ .Values.name | trunc 63 | trimSuffix "-" | quote }}
{{- end }}

{{- end }}

{{ include "labels_without_key_or_name" . | indent 2 }}

{{- end }}



{{- define "labels_without_key_or_name" }}
{{- if .Values.usingNewRecommendedLabels }}
# see: https://kubernetes.io/docs/concepts/overview/working-with-objects/common-labels/
app.kubernetes.io/version: {{ .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" | quote }}
app.kubernetes.io/component: {{ .Chart.Name | replace "+" "_" | trunc 63 | trimSuffix "-" | quote }}
app.kubernetes.io/created-by: "devops-nirvana"
app.kubernetes.io/managed-by: "helm"
{{ if .Values.labels -}}
{{ toYaml .Values.labels }}
{{- end }}

{{- else }}
chart: {{ include "chart" . | quote }}
release: {{ .Release.Name | quote }}
heritage: {{ .Release.Service | quote }}
helm_chart_author: "devops-nirvana"
generator: "helm"
{{ if .Values.labels -}}
{{ toYaml .Values.labels }}
{{- end }}

{{- end }}

{{- end }}


{{/*
Get our repo based on our environment name, but allow overriding if someone thinks they know better
*/}}
{{- define "get-repository" -}}
{{- required "A valid Values.image.repository is required!" .Values.image.repository | trimSuffix ":" -}}
{{- end -}}


{{/*
Get our release tag the best we can.
If we are customizing the image repository then this is not built internally, use image.tag or latest
If we are not, then use global image tag if set (from gitlab ci/cd) or fallback to image tag
*/}}
{{- define "get-release-tag" -}}

{{- if .Values.image.repository -}}

{{- if .Values.image.tag -}}
{{- .Values.image.tag -}}
{{- else if .Values.global.image.tag -}}
{{- .Values.global.image.tag -}}
{{- else -}}
latest
{{- end -}}

{{- else -}}

{{- if .Values.global.image.tag -}}
{{- .Values.global.image.tag -}}
{{- else if .Values.image.tag -}}
{{- .Values.image.tag -}}
{{- else -}}
no-image-tag-could-be-found
{{- end -}}

{{- end -}}

{{- end -}}


{{/*
Create the name of the ingress resource (used for legacy purposes and zero downtime for legacy)
*/}}
{{- define "ingress.name" -}}
{{- if .Values.ingress.name -}}
    {{ .Values.ingress.name }}
{{- else -}}
    {{ template "name" . }}
{{- end -}}
{{- end -}}


{{/*
Create the name of the ingress_secondary resource (used for legacy purposes and zero downtime for legacy)
*/}}
{{- define "ingress_secondary.name" -}}
{{- if .Values.ingress_secondary.name -}}
    {{ .Values.ingress_secondary.name }}
{{- else -}}
    {{ template "name" . }}
{{- end -}}
{{- end -}}

{{/*
Return the appropriate apiVersion for deployment.
*/}}
{{- define "deployment.apiVersion" -}}
  {{- print "apps/v1" -}}
{{- end -}}

{{/*
Return the appropriate apiVersion for daemonset.
*/}}
{{- define "daemonset.apiVersion" -}}
  {{- print "apps/v1" -}}
{{- end -}}

{{/*
Return the appropriate apiVersion for statefulset.
*/}}
{{- define "statefulset.apiVersion" -}}
  {{- print "apps/v1" -}}
{{- end -}}

{{/*
Return the appropriate apiVersion for networkpolicy.
*/}}
{{- define "networkPolicy.apiVersion" -}}
  {{- print "networking.k8s.io/v1" -}}
{{- end -}}

{{/*
Return the appropriate apiVersion for podsecuritypolicy.
*/}}
{{- define "podSecurityPolicy.apiVersion" -}}
  {{- print "policy/v1beta1" -}}
{{- end -}}

{{/*
Return the appropriate apiVersion for rbac.
*/}}
{{- define "rbac.apiVersion" -}}
  {{- if (default $.Capabilities "").APIVersions.Has "rbac.authorization.k8s.io/v1" }}
    {{- print "rbac.authorization.k8s.io/v1" -}}
  {{- else -}}
    {{- print "rbac.authorization.k8s.io/v1beta1" -}}
  {{- end -}}
{{- end -}}

{{/*
Return the appropriate apiVersion for cronjob.
*/}}
{{- define "cronjob.apiVersion" -}}
  {{ ternary "batch/v1" "batch/v1beta1" (.Capabilities.APIVersions.Has "batch/v1") }}
{{- end -}}


{{/*
Return the appropriate apiVersion for ingress.
*/}}
{{- define "ingress.apiVersion" -}}
  {{- if (default $.Capabilities "").APIVersions.Has "networking.k8s.io/v1" }}
    {{- print "networking.k8s.io/v1" -}}
  {{- else if .Capabilities.APIVersions.Has "networking.k8s.io/v1beta1" -}}
    {{- print "networking.k8s.io/v1beta1" -}}
  {{- else -}}
    {{- print "extensions/v1beta1" -}}
  {{- end -}}
{{- end -}}

{{/*
Return the appropriate apiVersion for poddisruptionbudget.
*/}}
{{- define "pdb.apiVersion" -}}
  {{- if (default $.Capabilities "").APIVersions.Has "policy/v1" }}
    {{- print "policy/v1" -}}
  {{- else -}}
    {{- print "policy/v1beta1" -}}
  {{- end -}}
{{- end -}}

{{/*
Return the appropriate apiVersion for horizontalpodautoscaler.
*/}}
{{- define "hpa.apiVersion" -}}
  {{ ternary "autoscaling/v2" "autoscaling/v2beta2" (.Capabilities.APIVersions.Has "autoscaling/v2") }}
{{- end }}

{{/*
Return if ingress is stable.
*/}}
{{- define "ingress.isStable" -}}
  {{- eq (include "ingress.apiVersion" .) "networking.k8s.io/v1" -}}
{{- end -}}

{{/*
Return if ingress supports ingressClassName.
*/}}
{{- define "ingress.supportsIngressClassName" -}}
  {{- or (eq (include "ingress.isStable" .) "true") (eq (include "ingress.apiVersion" .) "networking.k8s.io/v1beta1") -}}
{{- end -}}

{{/*
Return if ingress supports pathType.
*/}}
{{- define "ingress.supportsPathType" -}}
  {{- or (eq (include "ingress.isStable" .) "true") (eq (include "ingress.apiVersion" .) "networking.k8s.io/v1beta1") -}}
{{- end -}}
