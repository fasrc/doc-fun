"""
Link validator plugin for checking HTTP response codes of all links in documentation.
"""

import re
import time
import logging
import requests
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup

from ..plugins.analysis_base import AnalysisPlugin


class LinkValidator(AnalysisPlugin):
    """
    Validates all links in generated documentation by checking HTTP response codes.
    
    This plugin extracts all links from HTML documents and verifies they return
    valid HTTP responses (200-299 range). It handles timeouts, retries, and
    provides detailed reporting of broken links.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, config: Optional[Dict] = None, **kwargs):
        """
        Initialize the LinkValidator.
        
        Config options:
            timeout: Request timeout in seconds (default: 10)
            retries: Number of retry attempts for failed requests (default: 2)
            retry_delay: Delay between retries in seconds (default: 1)
            max_workers: Maximum number of concurrent requests (default: 5)
            check_internal: Check internal/anchor links (default: False)
            user_agent: User agent string for requests
            ignore_patterns: List of regex patterns to ignore
            validate_ssl: Verify SSL certificates (default: True)
        """
        super().__init__(logger, config, **kwargs)
        
        # Configuration
        self.timeout = self.config.get('timeout', 10)
        self.retries = self.config.get('retries', 2)
        self.retry_delay = self.config.get('retry_delay', 1)
        self.max_workers = self.config.get('max_workers', 5)
        self.check_internal = self.config.get('check_internal', False)
        self.validate_ssl = self.config.get('validate_ssl', True)
        self.report_format = self.config.get('report_format', 'markdown')
        
        # User agent for requests
        self.user_agent = self.config.get('user_agent', 
            'Mozilla/5.0 (compatible; DocGeneratorLinkValidator/1.0; +http://github.com/fasrc/doc-fun)')
        
        # Patterns to ignore (e.g., localhost, internal networks)
        default_ignore = [
            r'^mailto:',
            r'^tel:',
            r'^ftp:',
            r'^javascript:',
            r'^#',  # Anchor links
            r'^localhost',
            r'^127\.0\.0\.1',
            r'^192\.168\.',
            r'^10\.',
            r'^172\.(1[6-9]|2[0-9]|3[01])\.'
        ]
        self.ignore_patterns = self.config.get('ignore_patterns', default_ignore)
        
        # Request session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
    
    def get_name(self) -> str:
        """Return the plugin name."""
        return 'link_validator'
    
    def analyze(self, documents: List[Dict[str, Any]], topic: str) -> Dict[str, Any]:
        """
        Extract and validate all links from the documents.
        
        Args:
            documents: List of document dictionaries with 'path' and 'content'
            topic: The topic used for generation
            
        Returns:
            Dictionary containing:
                - total_links: Total number of links found
                - unique_links: Number of unique URLs
                - valid_links: Links that returned 200-299 status
                - broken_links: Links that failed or returned error status
                - skipped_links: Links that were ignored based on patterns
                - link_details: Detailed information for each link
        """
        all_links = {}  # {url: {documents: [], status: None, message: ''}}
        
        # Extract links from all documents
        for doc_idx, doc in enumerate(documents):
            content = doc.get('content', '')
            path = doc.get('path', f'document_{doc_idx}')
            
            if not content:
                self.logger.warning(f"Document {doc_idx} has no content")
                continue
            
            # Extract links from this document
            links = self._extract_links(content)
            
            for link in links:
                if link not in all_links:
                    all_links[link] = {
                        'documents': [],
                        'status': None,
                        'status_code': None,
                        'message': '',
                        'response_time': None
                    }
                all_links[link]['documents'].append(path)
        
        # Filter out ignored patterns
        links_to_check = []
        skipped_links = []
        
        for url in all_links.keys():
            if self._should_ignore(url):
                all_links[url]['status'] = 'skipped'
                all_links[url]['message'] = 'Matched ignore pattern'
                skipped_links.append(url)
            else:
                links_to_check.append(url)
        
        self.logger.info(f"Found {len(all_links)} total links, checking {len(links_to_check)}, skipping {len(skipped_links)}")
        
        # Validate links concurrently
        if links_to_check:
            self._validate_links_concurrent(links_to_check, all_links)
        
        # Categorize results
        valid_links = []
        broken_links = []
        
        for url, details in all_links.items():
            if details['status'] == 'valid':
                valid_links.append(url)
            elif details['status'] == 'broken':
                broken_links.append(url)
        
        return {
            'total_links': len(all_links),
            'unique_links': len(all_links),
            'valid_links': valid_links,
            'broken_links': broken_links,
            'skipped_links': skipped_links,
            'link_details': all_links,
            'summary': {
                'total': len(all_links),
                'valid': len(valid_links),
                'broken': len(broken_links),
                'skipped': len(skipped_links),
                'success_rate': (len(valid_links) / len(links_to_check) * 100) if links_to_check else 100
            }
        }
    
    def _extract_links(self, html_content: str) -> List[str]:
        """Extract all links from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        
        # Find all anchor tags
        for tag in soup.find_all('a', href=True):
            href = tag['href'].strip()
            if href:
                links.append(href)
        
        # Also find links in src attributes (images, scripts)
        for tag in soup.find_all(['img', 'script', 'link'], src=True):
            src = tag.get('src', '').strip()
            if src and src.startswith('http'):
                links.append(src)
        
        # Find links in link tags (CSS, etc.)
        for tag in soup.find_all('link', href=True):
            href = tag['href'].strip()
            if href and href.startswith('http'):
                links.append(href)
        
        return links
    
    def _should_ignore(self, url: str) -> bool:
        """Check if a URL should be ignored based on patterns."""
        for pattern in self.ignore_patterns:
            if re.match(pattern, url, re.IGNORECASE):
                return True
        
        # Skip internal anchors unless configured to check them
        if not self.check_internal and url.startswith('#'):
            return True
        
        return False
    
    def _validate_links_concurrent(self, urls: List[str], all_links: Dict):
        """Validate multiple links concurrently."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all validation tasks
            future_to_url = {
                executor.submit(self._validate_single_link, url): url 
                for url in urls
            }
            
            # Process completed tasks
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    status, status_code, message, response_time = future.result()
                    all_links[url]['status'] = status
                    all_links[url]['status_code'] = status_code
                    all_links[url]['message'] = message
                    all_links[url]['response_time'] = response_time
                except Exception as e:
                    all_links[url]['status'] = 'error'
                    all_links[url]['message'] = f'Validation error: {str(e)}'
                    self.logger.error(f"Error validating {url}: {e}")
    
    def _validate_single_link(self, url: str) -> Tuple[str, Optional[int], str, Optional[float]]:
        """
        Validate a single link with retries.
        
        Returns:
            Tuple of (status, status_code, message, response_time)
        """
        for attempt in range(self.retries + 1):
            try:
                start_time = time.time()
                
                # Make the request
                response = self.session.head(
                    url, 
                    timeout=self.timeout,
                    allow_redirects=True,
                    verify=self.validate_ssl
                )
                
                response_time = time.time() - start_time
                
                # Some servers don't support HEAD, try GET
                if response.status_code == 405:  # Method Not Allowed
                    response = self.session.get(
                        url,
                        timeout=self.timeout,
                        allow_redirects=True,
                        verify=self.validate_ssl,
                        stream=True  # Don't download content
                    )
                    response.close()
                
                # Check status code
                if 200 <= response.status_code < 300:
                    return ('valid', response.status_code, 'OK', response_time)
                elif response.status_code == 404:
                    return ('broken', response.status_code, 'Not Found', response_time)
                elif response.status_code == 403:
                    return ('broken', response.status_code, 'Forbidden', response_time)
                elif response.status_code >= 500:
                    if attempt < self.retries:
                        time.sleep(self.retry_delay)
                        continue
                    return ('broken', response.status_code, f'Server Error: {response.status_code}', response_time)
                else:
                    return ('broken', response.status_code, f'HTTP {response.status_code}', response_time)
                
            except requests.exceptions.Timeout:
                if attempt < self.retries:
                    time.sleep(self.retry_delay)
                    continue
                return ('broken', None, 'Timeout', None)
                
            except requests.exceptions.SSLError:
                return ('broken', None, 'SSL Certificate Error', None)
                
            except requests.exceptions.ConnectionError:
                if attempt < self.retries:
                    time.sleep(self.retry_delay)
                    continue
                return ('broken', None, 'Connection Failed', None)
                
            except requests.exceptions.TooManyRedirects:
                return ('broken', None, 'Too Many Redirects', None)
                
            except Exception as e:
                if attempt < self.retries:
                    time.sleep(self.retry_delay)
                    continue
                return ('broken', None, f'Error: {str(e)}', None)
        
        return ('broken', None, 'Max retries exceeded', None)
    
    def generate_report(self, analysis_results: Dict[str, Any], topic: str) -> str:
        """
        Generate a markdown report of link validation results.
        
        Args:
            analysis_results: Results from analyze() method
            topic: The topic used for generation
            
        Returns:
            Markdown-formatted report
        """
        report_lines = [
            f'# Link Validation Report: {topic}',
            '',
            '## Summary',
            ''
        ]
        
        summary = analysis_results.get('summary', {})
        report_lines.extend([
            f"- **Total Links Found:** {summary.get('total', 0)}",
            f"- **Valid Links:** {summary.get('valid', 0)} ✅",
            f"- **Broken Links:** {summary.get('broken', 0)} ❌",
            f"- **Skipped Links:** {summary.get('skipped', 0)} ⏭️",
            f"- **Success Rate:** {summary.get('success_rate', 100):.1f}%",
            ''
        ])
        
        # Broken links section
        broken_links = analysis_results.get('broken_links', [])
        if broken_links:
            report_lines.extend([
                '## ❌ Broken Links',
                '',
                'These links need attention:',
                ''
            ])
            
            link_details = analysis_results.get('link_details', {})
            for url in broken_links:
                details = link_details.get(url, {})
                status_code = details.get('status_code', 'N/A')
                message = details.get('message', 'Unknown error')
                docs = details.get('documents', [])
                doc_names = [Path(d).name if isinstance(d, str) else f"Doc {d}" for d in docs[:3]]
                
                report_lines.append(f"### `{url}`")
                report_lines.append(f"- **Status:** {status_code} - {message}")
                report_lines.append(f"- **Found in:** {', '.join(doc_names)}")
                if len(docs) > 3:
                    report_lines.append(f"  (and {len(docs) - 3} more documents)")
                report_lines.append('')
        
        # Valid links summary (abbreviated)
        valid_links = analysis_results.get('valid_links', [])
        if valid_links:
            report_lines.extend([
                '## ✅ Valid Links',
                '',
                f'Successfully validated {len(valid_links)} links:',
                ''
            ])
            
            # Show first 10 valid links
            for url in valid_links[:10]:
                report_lines.append(f"- `{url}`")
            
            if len(valid_links) > 10:
                report_lines.append(f"- ... and {len(valid_links) - 10} more")
            
            report_lines.append('')
        
        # Performance metrics
        link_details = analysis_results.get('link_details', {})
        response_times = [d['response_time'] for d in link_details.values() 
                         if d.get('response_time') is not None]
        
        if response_times:
            avg_response = sum(response_times) / len(response_times)
            max_response = max(response_times)
            min_response = min(response_times)
            
            report_lines.extend([
                '## Performance Metrics',
                '',
                f"- **Average Response Time:** {avg_response:.2f}s",
                f"- **Fastest Response:** {min_response:.2f}s",
                f"- **Slowest Response:** {max_response:.2f}s",
                ''
            ])
        
        # Skipped links summary
        skipped_links = analysis_results.get('skipped_links', [])
        if skipped_links:
            report_lines.extend([
                '## ⏭️ Skipped Links',
                '',
                f'Skipped {len(skipped_links)} links based on ignore patterns:',
                ''
            ])
            
            # Group by type
            mailto = [l for l in skipped_links if l.startswith('mailto:')]
            anchors = [l for l in skipped_links if l.startswith('#')]
            others = [l for l in skipped_links if not l.startswith('mailto:') and not l.startswith('#')]
            
            if mailto:
                report_lines.append(f"- Email links: {len(mailto)}")
            if anchors:
                report_lines.append(f"- Anchor links: {len(anchors)}")
            if others:
                report_lines.append(f"- Other ignored patterns: {len(others)}")
            
            report_lines.append('')
        
        return '\n'.join(report_lines)
    
    def generate_html_report(self, analysis_results: Dict[str, Any], topic: str) -> str:
        """
        Generate an HTML report of link validation results.
        
        Args:
            analysis_results: Results from analyze() method
            topic: The topic used for generation
            
        Returns:
            HTML-formatted report
        """
        html_parts = [
            '<!DOCTYPE html>',
            '<html lang="en">',
            '<head>',
            '<meta charset="UTF-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
            f'<title>Link Validation Report: {topic}</title>',
            '<style>',
            'body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; ',
            '       line-height: 1.6; max-width: 1000px; margin: 0 auto; padding: 20px; }',
            'h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }',
            'h2 { color: #34495e; margin-top: 30px; }',
            '.summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); ',
            '                gap: 15px; margin: 20px 0; }',
            '.summary-card { background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }',
            '.summary-value { font-size: 28px; font-weight: bold; }',
            '.summary-label { color: #7f8c8d; font-size: 12px; margin-top: 5px; }',
            '.valid { color: #27ae60; }',
            '.broken { color: #e74c3c; }',
            '.skipped { color: #95a5a6; }',
            '.success-rate { color: #3498db; }',
            '.link-card { background: white; border: 1px solid #ecf0f1; padding: 15px; ',
            '             margin: 10px 0; border-radius: 8px; border-left: 4px solid #e74c3c; }',
            '.link-url { font-family: monospace; font-size: 14px; word-break: break-all; ',
            '            margin-bottom: 10px; color: #2c3e50; }',
            '.link-status { font-weight: bold; color: #e74c3c; }',
            '.link-docs { color: #7f8c8d; font-size: 14px; }',
            '.performance-stats { background: #ecf0f1; padding: 15px; border-radius: 8px; margin: 20px 0; }',
            'table { width: 100%; border-collapse: collapse; margin: 20px 0; }',
            'th { background: #3498db; color: white; padding: 10px; text-align: left; }',
            'td { padding: 10px; border-bottom: 1px solid #ecf0f1; }',
            'code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-size: 14px; }',
            '</style>',
            '</head>',
            '<body>',
            f'<h1>Link Validation Report: {topic}</h1>'
        ]
        
        # Summary section
        summary = analysis_results.get('summary', {})
        html_parts.append('<h2>Summary</h2>')
        html_parts.append('<div class="summary-grid">')
        html_parts.append(f'<div class="summary-card"><div class="summary-value">{summary.get("total", 0)}</div><div class="summary-label">Total Links</div></div>')
        html_parts.append(f'<div class="summary-card"><div class="summary-value valid">{summary.get("valid", 0)}</div><div class="summary-label">✅ Valid</div></div>')
        html_parts.append(f'<div class="summary-card"><div class="summary-value broken">{summary.get("broken", 0)}</div><div class="summary-label">❌ Broken</div></div>')
        html_parts.append(f'<div class="summary-card"><div class="summary-value skipped">{summary.get("skipped", 0)}</div><div class="summary-label">⏭️ Skipped</div></div>')
        html_parts.append(f'<div class="summary-card"><div class="summary-value success-rate">{summary.get("success_rate", 100):.1f}%</div><div class="summary-label">Success Rate</div></div>')
        html_parts.append('</div>')
        
        # Broken links section
        broken_links = analysis_results.get('broken_links', [])
        if broken_links:
            html_parts.append('<h2>❌ Broken Links</h2>')
            html_parts.append('<p>These links need attention:</p>')
            
            link_details = analysis_results.get('link_details', {})
            for url in broken_links:
                details = link_details.get(url, {})
                status_code = details.get('status_code', 'N/A')
                message = details.get('message', 'Unknown error')
                docs = details.get('documents', [])
                doc_names = [Path(d).name if isinstance(d, str) else f"Doc {d}" for d in docs]
                
                html_parts.append('<div class="link-card">')
                html_parts.append(f'<div class="link-url">{url}</div>')
                html_parts.append(f'<div class="link-status">Status: {status_code} - {message}</div>')
                html_parts.append(f'<div class="link-docs">Found in: {", ".join(doc_names[:3])}')
                if len(docs) > 3:
                    html_parts.append(f' (and {len(docs) - 3} more)')
                html_parts.append('</div>')
                html_parts.append('</div>')
        
        # Valid links summary
        valid_links = analysis_results.get('valid_links', [])
        if valid_links:
            html_parts.append('<h2>✅ Valid Links</h2>')
            html_parts.append(f'<p>Successfully validated {len(valid_links)} links:</p>')
            html_parts.append('<table>')
            html_parts.append('<tr><th>URL</th><th>Response Time</th></tr>')
            
            link_details = analysis_results.get('link_details', {})
            # Show first 10 valid links
            for url in valid_links[:10]:
                details = link_details.get(url, {})
                response_time = details.get('response_time')
                time_str = f'{response_time:.2f}s' if response_time else 'N/A'
                html_parts.append(f'<tr><td><code>{url}</code></td><td>{time_str}</td></tr>')
            
            if len(valid_links) > 10:
                html_parts.append(f'<tr><td colspan="2"><em>... and {len(valid_links) - 10} more valid links</em></td></tr>')
            
            html_parts.append('</table>')
        
        # Performance metrics
        link_details = analysis_results.get('link_details', {})
        response_times = [d['response_time'] for d in link_details.values() 
                         if d.get('response_time') is not None]
        
        if response_times:
            avg_response = sum(response_times) / len(response_times)
            max_response = max(response_times)
            min_response = min(response_times)
            
            html_parts.append('<h2>Performance Metrics</h2>')
            html_parts.append('<div class="performance-stats">')
            html_parts.append(f'<p><strong>Average Response Time:</strong> {avg_response:.2f}s</p>')
            html_parts.append(f'<p><strong>Fastest Response:</strong> {min_response:.2f}s</p>')
            html_parts.append(f'<p><strong>Slowest Response:</strong> {max_response:.2f}s</p>')
            html_parts.append('</div>')
        
        # Skipped links summary
        skipped_links = analysis_results.get('skipped_links', [])
        if skipped_links:
            html_parts.append('<h2>⏭️ Skipped Links</h2>')
            html_parts.append(f'<p>Skipped {len(skipped_links)} links based on ignore patterns:</p>')
            
            # Group by type
            mailto = [l for l in skipped_links if l.startswith('mailto:')]
            anchors = [l for l in skipped_links if l.startswith('#')]
            others = [l for l in skipped_links if not l.startswith('mailto:') and not l.startswith('#')]
            
            html_parts.append('<ul>')
            if mailto:
                html_parts.append(f'<li>Email links: {len(mailto)}</li>')
            if anchors:
                html_parts.append(f'<li>Anchor links: {len(anchors)}</li>')
            if others:
                html_parts.append(f'<li>Other ignored patterns: {len(others)}</li>')
            html_parts.append('</ul>')
        
        html_parts.extend([
            '</body>',
            '</html>'
        ])
        
        return '\n'.join(html_parts)
    
    def save_artifacts(self, results: Dict[str, Any], output_dir: Path, topic: str) -> List[Path]:
        """
        Save link validation report to file.
        
        Args:
            results: Analysis results
            output_dir: Directory to save artifacts
            topic: The topic used for generation
            
        Returns:
            List of saved file paths
        """
        saved_files = []
        safe_topic = self.sanitize_filename(topic)
        
        # Save report in configured format
        if self.report_format == 'html':
            report = self.generate_html_report(results, topic)
            report_path = output_dir / f'{safe_topic}_link_validation.html'
        else:  # Default to markdown
            report = self.generate_report(results, topic)
            report_path = output_dir / f'{safe_topic}_link_validation.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        saved_files.append(report_path)
        self.logger.info(f"Saved link validation report to {report_path}")
        
        # Save detailed results as text file for broken links only
        broken_links = results.get('broken_links', [])
        if broken_links:
            broken_path = output_dir / f'{safe_topic}_broken_links.txt'
            with open(broken_path, 'w', encoding='utf-8') as f:
                f.write(f"Broken Links Report for: {topic}\n")
                f.write("=" * 50 + "\n\n")
                
                link_details = results.get('link_details', {})
                for url in broken_links:
                    details = link_details.get(url, {})
                    f.write(f"URL: {url}\n")
                    f.write(f"Status: {details.get('status_code', 'N/A')} - {details.get('message', 'Unknown')}\n")
                    f.write(f"Found in: {', '.join(details.get('documents', []))}\n")
                    f.write("-" * 30 + "\n")
            
            saved_files.append(broken_path)
            self.logger.info(f"Saved broken links list to {broken_path}")
        
        return saved_files
    
    def get_priority(self) -> int:
        """Link validation should run after compilation and reporting."""
        return 50
    
    def get_supported_formats(self) -> List[str]:
        """Return supported output formats."""
        return ['markdown', 'text']
    
    def __del__(self):
        """Clean up the session when the plugin is destroyed."""
        if hasattr(self, 'session'):
            self.session.close()