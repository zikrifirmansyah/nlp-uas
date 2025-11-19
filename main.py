from flask import Flask, render_template_string, request, redirect, url_for
import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure necessary NLTK data is available
try:
		nltk.data.find('tokenizers/punkt')
except LookupError:
		nltk.download('punkt', quiet=True)

try:
		nltk.data.find('corpora/stopwords')
except LookupError:
		nltk.download('stopwords', quiet=True)

app = Flask(__name__)

# Fallback Indonesian stopwords if not available via NLTK
DEFAULT_IND_STOPWORDS = {
		'yang','dan','di','ke','dari','ini','itu','untuk','pada','dengan','atau','sebagai',
		'adalah','saya','kamu','kita','dia','ya','nya','dengan','oleh','akan','tidak','tida',
		'ada','ini','itu','yg','nih','ga','gak','yg','supaya','biar','masa'
}

def get_stopwords(lang='indonesian'):
		try:
				sw = set(stopwords.words(lang))
				if len(sw) > 10:
						return sw
		except Exception:
				pass
		return DEFAULT_IND_STOPWORDS

STOPWORDS = get_stopwords('indonesian')

EMOJI_PATTERN = re.compile("["
													 "\U0001F600-\U0001F64F"  # emoticons
													 "\U0001F300-\U0001F5FF"  # symbols & pictographs
													 "\U0001F680-\U0001F6FF"  # transport & map symbols
													 "\U0001F1E0-\U0001F1FF"  # flags (iOS)
													 "\u2600-\u26FF\u2700-\u27BF]+", flags=re.UNICODE)

def clean_text(text: str) -> str:
		if pd.isna(text):
				return ""
		# convert to str
		text = str(text)
		# remove urls
		text = re.sub(r'http\S+|www\.\S+', ' ', text)
		# remove emojis
		text = EMOJI_PATTERN.sub(' ', text)
		# remove mentions and hashtags
		text = re.sub(r'[@#]\w+', ' ', text)
		# remove non-letter characters (keep unicode letters)
		text = re.sub(r'[^\w\s]', ' ', text)
		# remove digits
		text = re.sub(r'\d+', ' ', text)
		# collapse whitespace
		text = re.sub(r'\s+', ' ', text).strip()
		return text.lower()

def tokenize_and_filter(text: str):
		if not text:
				return []
		try:
				tokens = word_tokenize(text)
		except Exception:
				# fallback simple split
				tokens = text.split()
		# normalize tokens
		tokens = [t.lower() for t in tokens if len(t) > 1]
		# remove stopwords
		tokens = [t for t in tokens if t not in STOPWORDS]
		return tokens


def df_to_html_with_links(df: pd.DataFrame, max_rows: int | None = None) -> str:
	"""Return HTML for DataFrame where any column containing 'url' is rendered as a clickable link.
	If max_rows is provided, use head(max_rows).
	"""
	if max_rows:
		df = df.head(max_rows)
	else:
		df = df.copy()
	# work on a copy to avoid mutating original
	df = df.copy()
	for col in df.columns:
		if 'url' in str(col).lower():
			# replace with anchor tags; keep text as the url itself per request
			df[col] = df[col].fillna('').apply(lambda u: f'<a href="{u}" target="_blank" rel="noopener noreferrer">url</a>' if str(u).strip() else '')
	# render without escaping so anchors are active
	return df.to_html(classes='table table-sm table-striped', index=False, escape=False)

INDEX_HTML = '''
<!doctype html>
<html lang="en">
	<head>
		<meta charset="utf-8">
		<meta name="viewport" content="width=device-width, initial-scale=1">
		<title>NLP Preprocessing</title>
		<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
	</head>
	<body class="bg-light">
		<div class="container py-4">
			<h1 class="mb-3">NLP Preprocessing - Simple Web UI</h1>
			<p class="text-muted">Pilih kolom teks dari dataset `dataNew.csv` untuk memproses (comments, caption, atau kolom lain).</p>

			{% if error %}
				<div class="alert alert-danger">{{ error }}</div>
			{% endif %}

			<form method="post" action="/preprocess" class="row g-3 mb-4">
				<div class="col-md-4">
					<label class="form-label">CSV Path</label>
					<input name="csv_path" class="form-control" value="dataNew.csv">
				</div>
				<div class="col-md-4">
					<label class="form-label">Text Column</label>
					<input name="text_col" class="form-control" value="comments">
				</div>
				<div class="col-md-2">
					<label class="form-label">Rows to show</label>
					<input name="n_rows" type="number" min="1" max="100000" class="form-control" value="{{ rows_to_show or 20 }}">
				</div>
				<div class="col-md-2 align-self-end">
					<button class="btn btn-primary w-100">Preprocess</button>
				</div>
			</form>

			<div class="mb-3">
				<form method="get" action="/show_all" class="d-inline">
					<button class="btn btn-outline-secondary">Show All Data</button>
				</form>
			</div>

			{% if table_html %}
				<h5>Data Preview (original)</h5>
				<div class="table-responsive">{{ table_html | safe }}</div>
			{% endif %}

			<hr>
			<footer class="text-muted small">If NLTK data is missing, server will attempt to download it once.</footer>
		</div>
	</body>
</html>
'''

RESULT_HTML = '''
<!doctype html>
<html lang="en">
	<head>
		<meta charset="utf-8">
		<meta name="viewport" content="width=device-width, initial-scale=1">
		<title>Preprocessing Result</title>
		<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
	</head>
	<body class="bg-light">
		<div class="container py-4">
			<h1 class="mb-3">Preprocessing Result</h1>
			<a href="/" class="btn btn-secondary mb-3">&larr; Back</a>

			<h5>Processed (first {{ n_rows }} rows)</h5>
			<div class="d-flex justify-content-between align-items-center mb-2">
				<div>Page {{ page }} / {{ total_pages }}</div>
				<div>
					<form method="get" action="/preprocess" class="d-inline">
						<input type="hidden" name="csv_path" value="{{ csv_path }}">
						<input type="hidden" name="text_col" value="{{ text_col }}">
						<input type="hidden" name="page_size" value="{{ page_size }}">
						<button name="page" value="{{ 1 }}" class="btn btn-sm btn-outline-secondary">First</button>
						{% if page > 1 %}
						<button name="page" value="{{ page-1 }}" class="btn btn-sm btn-outline-secondary">Prev</button>
						{% endif %}
						{% if page < total_pages %}
						<button name="page" value="{{ page+1 }}" class="btn btn-sm btn-outline-secondary">Next</button>
						{% endif %}
						<button name="page" value="{{ total_pages }}" class="btn btn-sm btn-outline-secondary">Last</button>
					</form>
				</div>
			</div>
			<div class="table-responsive">{{ result_table | safe }}</div>

			<hr>
			<h6>Notes</h6>
			<ul>
				<li>Cleaning steps: remove URLs, emojis, mentions, punctuation, digits.</li>
				<li>Tokenization: NLTK word_tokenize (fallback to split on failure).</li>
				<li>Stopwords: NLTK Indonesian stopwords if available, otherwise fallback list.</li>
			</ul>
		</div>
	</body>
</html>
'''



@app.route('/', methods=['GET'])
def index():
	csv_path = os.path.join(os.getcwd(), 'dataNew.csv')
	table_html = None
	error = None
	if os.path.exists(csv_path):
		try:
			df = pd.read_csv(csv_path, nrows=20)
			table_html = df_to_html_with_links(df, max_rows=20)
		except Exception as e:
			error = f"Error reading CSV: {e}"
	else:
		error = f"File not found: {csv_path}"

	return render_template_string(INDEX_HTML, table_html=table_html, error=error)



@app.route('/show_all', methods=['GET'])
def show_all():
	csv_path = os.path.join(os.getcwd(), 'dataNew.csv')
	# ensure file exists
	if not os.path.exists(csv_path):
		return render_template_string(INDEX_HTML, table_html=None, error=f"File not found: {csv_path}")
	try:
		df = pd.read_csv(csv_path)
		table_html = df_to_html_with_links(df)
		# when showing all data, update the form's rows_to_show to the number of rows
		rows = int(df.shape[0]) if hasattr(df, 'shape') else None
		return render_template_string(INDEX_HTML, table_html=table_html, error=None, rows_to_show=rows)
	except Exception as e:
		return render_template_string(INDEX_HTML, table_html=None, error=f"Error reading CSV: {e}")



@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
	# POST: receive form -> redirect to GET with query params for pagination
	if request.method == 'POST':
		csv_path = request.form.get('csv_path', 'dataNew.csv')
		text_col = request.form.get('text_col', 'comments')
		try:
			page_size = int(request.form.get('n_rows', 20))
		except Exception:
			page_size = 20
		# start at page 1
		return redirect(url_for('preprocess', csv_path=csv_path, text_col=text_col, page=1, page_size=page_size))

	# GET: show paginated preprocessing results
	csv_path = request.args.get('csv_path', 'dataNew.csv')
	text_col = request.args.get('text_col', 'comments')
	try:
		page = int(request.args.get('page', 1))
	except Exception:
		page = 1
	try:
		page_size = int(request.args.get('page_size', request.args.get('n_rows', 20)))
	except Exception:
		page_size = 20

	if not os.path.exists(csv_path):
		return render_template_string(INDEX_HTML, table_html=None, error=f"CSV not found: {csv_path}")

	try:
		df = pd.read_csv(csv_path)
	except Exception as e:
		return render_template_string(INDEX_HTML, table_html=None, error=f"Error reading CSV: {e}")

	if text_col not in df.columns:
		return render_template_string(INDEX_HTML, table_html=None, error=f"Column '{text_col}' not found in CSV. Available columns: {', '.join(df.columns)}")

	# Apply preprocessing for entire column, then paginate
	texts = df[text_col].astype(str).fillna('')

	def process_segments(text):
		# split on pipe | to get segments, strip spaces, ignore empty
		segments = [s.strip() for s in str(text).split('|')]
		segments = [s for s in segments if s]
		if not segments:
			return ('', '')
		cleaned_segs = [clean_text(s) for s in segments]
		token_segs = [tokenize_and_filter(s) for s in cleaned_segs]
		# join cleaned segments with ' | ' and tokens per segment joined by spaces
		cleaned_join = ' | '.join(cleaned_segs)
		tokens_join = ' | '.join([' '.join(t) for t in token_segs])
		return (cleaned_join, tokens_join)

	processed = texts.map(process_segments)
	cleaned = processed.map(lambda x: x[0])
	tokens = processed.map(lambda x: x[1])

	result_df = pd.DataFrame({
		'original': texts,
		'cleaned_segments': cleaned,
		'tokens_per_segment': tokens
	})

	total = int(result_df.shape[0])
	total_pages = max(1, (total + page_size - 1) // page_size)
	if page < 1:
		page = 1
	if page > total_pages:
		page = total_pages
	start = (page - 1) * page_size
	end = start + page_size
	slice_df = result_df.iloc[start:end].copy()

	# If text_col looks like URL, convert original column to links
	if 'url' in text_col.lower():
		slice_df['original'] = slice_df['original'].fillna('').apply(lambda u: f'<a href="{u}" target="_blank" rel="noopener noreferrer">{u}</a>' if str(u).strip() else '')
		result_table = slice_df.to_html(classes='table table-sm table-striped', index=False, escape=False)
	else:
		result_table = slice_df.to_html(classes='table table-sm table-striped', index=False, escape=True)

	# render result with pagination context
	return render_template_string(RESULT_HTML,
		result_table=result_table,
		n_rows=page_size,
		page=page,
		total_pages=total_pages,
		csv_path=csv_path,
		text_col=text_col,
		page_size=page_size)


if __name__ == '__main__':
		# Use port 5000 by default. Run with `python main.py`.
		app.run(host='0.0.0.0', port=5500, debug=True)

