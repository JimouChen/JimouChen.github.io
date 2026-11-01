import { getCollection, type CollectionEntry } from 'astro:content';

export type Post = CollectionEntry<'posts'>;

export async function getPublishedPosts(): Promise<Post[]> {
  const posts = await getCollection('posts', ({ data }) => !data.draft);
  return posts.sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf());
}

export function getPostUrl(post: Post): string {
  return `/blog/${post.id}/`;
}

export function getReadingMinutes(post: Post): number {
  const body = post.body ?? '';
  const chineseCharacters = body.match(/[\u3400-\u9fff]/g)?.length ?? 0;
  const latinWords = body.replace(/[\u3400-\u9fff]/g, ' ').match(/[\p{L}\p{N}]+/gu)?.length ?? 0;
  return Math.max(1, Math.ceil(chineseCharacters / 350 + latinWords / 200));
}

export function getCategories(posts: Post[]): Array<{ name: string; count: number }> {
  const counts = new Map<string, number>();
  posts.forEach((post) => counts.set(post.data.category, (counts.get(post.data.category) ?? 0) + 1));
  return [...counts.entries()].map(([name, count]) => ({ name, count })).sort((a, b) => a.name.localeCompare(b.name));
}

export function getTags(posts: Post[]): Array<{ name: string; count: number }> {
  const counts = new Map<string, number>();
  posts.flatMap((post) => post.data.tags).forEach((tag) => counts.set(tag, (counts.get(tag) ?? 0) + 1));
  return [...counts.entries()].map(([name, count]) => ({ name, count })).sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
}

export function getAdjacentPosts(posts: Post[], current: Post) {
  const index = posts.findIndex((post) => post.id === current.id);
  return {
    newer: index > 0 ? posts[index - 1] : undefined,
    older: index >= 0 ? posts[index + 1] : undefined,
  };
}

export function getRelatedPosts(posts: Post[], current: Post, limit = 3): Post[] {
  return posts
    .filter((post) => post.id !== current.id)
    .map((post) => ({
      post,
      score: (post.data.category === current.data.category ? 3 : 0)
        + post.data.tags.filter((tag) => current.data.tags.includes(tag)).length,
    }))
    .filter(({ score }) => score > 0)
    .sort((a, b) => b.score - a.score || b.post.data.pubDate.valueOf() - a.post.data.pubDate.valueOf())
    .slice(0, limit)
    .map(({ post }) => post);
}
