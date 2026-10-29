"""Reader chrome defaults; authors can supply labels for any document language."""
from typing import Annotated, Literal
from pydantic import Field

Label = Literal['skip', 'report', 'presentation', 'contents', 'present', 'read', 'print',
                'sources', 'no_sources', 'source', 'data_table', 'view_data', 'speaker_notes',
                'previous_slide', 'next_slide', 'no_records', 'no_values', 'missing', 'reference']
LabelText = Annotated[str, Field(min_length=1, max_length=160)]

EN = dict(zip(Label.__args__, [
    'Skip to content', 'Report', 'Presentation', 'Contents', 'Present', 'Read', 'Print / PDF',
    'Sources', 'No external data sources.', 'Source', 'Data table', 'View data', 'Speaker notes',
    'Previous slide', 'Next slide', 'No records match this selection.', 'No observed numeric values.',
    'Not available', 'Supporting visual resource']))
TRANSLATIONS = {
    'es': dict(zip(Label.__args__, [
        'Saltar al contenido', 'Informe', 'Presentación', 'Contenido', 'Presentar', 'Leer', 'Imprimir / PDF',
        'Fuentes', 'Sin fuentes de datos externas.', 'Fuente', 'Tabla de datos', 'Ver datos', 'Notas del presentador',
        'Diapositiva anterior', 'Diapositiva siguiente', 'Ningún registro coincide con esta selección.',
        'No hay valores numéricos observados.', 'No disponible', 'Recurso visual de apoyo'])),
    'fr': dict(zip(Label.__args__, [
        'Aller au contenu', 'Rapport', 'Présentation', 'Sommaire', 'Présenter', 'Lire', 'Imprimer / PDF',
        'Sources', 'Aucune source de données externe.', 'Source', 'Tableau de données', 'Voir les données',
        'Notes de présentation', 'Diapositive précédente', 'Diapositive suivante',
        'Aucun enregistrement ne correspond à cette sélection.', 'Aucune valeur numérique observée.',
        'Non disponible', 'Ressource visuelle complémentaire']))}


def reader_labels(language, overrides):
    return EN | TRANSLATIONS.get(language.lower().split('-')[0], {}) | overrides
