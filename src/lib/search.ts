export interface SearchResult {
  url: string;
  title: string;
  excerpt: string;
}

interface PagefindResult {
  data: () => Promise<{ url: string; meta?: { title?: string }; excerpt: string }>;
}

interface PagefindApi {
  init: () => Promise<void>;
  search: (query: string) => Promise<{ results: PagefindResult[] }>;
}

let pagefindPromise: Promise<PagefindApi> | undefined;

async function loadPagefind(): Promise<PagefindApi> {
  if (!pagefindPromise) {
    pagefindPromise = (async () => {
      const pagefindPath = '/pagefind/pagefind.js';
      const pagefind = await import(/* @vite-ignore */ pagefindPath) as PagefindApi;
      await pagefind.init();
      return pagefind;
    })();
  }
  return pagefindPromise;
}

export async function searchPosts(query: string): Promise<SearchResult[]> {
  const pagefind = await loadPagefind();
  const search = await pagefind.search(query);
  const results = await Promise.all(search.results.slice(0, 20).map((result) => result.data()));
  return results.map((result) => ({
    url: result.url,
    title: result.meta?.title ?? '未命名文章',
    excerpt: result.excerpt,
  }));
}
