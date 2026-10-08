import { FileCourseLibrary, LibraryError, libraryId } from '@/lib/course-library/server';
import type { AppDocument } from '@/lib/document-store/persistence-types';

export const runtime = 'nodejs';
const library = new FileCourseLibrary();
const LOCAL_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]']);

function guard(request: Request) {
  const url = new URL(request.url);
  // Next may normalize request.url to localhost even when the browser used
  // 127.0.0.1. Host is the browser-facing authority; never trust forwarded-host.
  const authority = new URL(`${url.protocol}//${request.headers.get('host') || url.host}`);
  if (!LOCAL_HOSTS.has(authority.hostname) || authority.username || authority.password)
    throw new LibraryError('The disk library is available only on localhost', 403);
  const origin = request.headers.get('origin');
  if (
    (origin && origin !== authority.origin) ||
    request.headers.get('sec-fetch-site') === 'cross-site'
  ) {
    throw new LibraryError('Cross-origin library access refused', 403);
  }
}
function failure(error: unknown) {
  console.error('Course library request failed:', error);
  return Response.json(
    {
      error:
        error instanceof LibraryError
          ? error.message
          : 'Could not access the course library. Check server storage and retry.',
    },
    { status: error instanceof LibraryError ? error.status : 500 },
  );
}
export async function GET(request: Request) {
  try {
    guard(request);
    const query = new URL(request.url).searchParams;
    if (query.has('asset')) {
      const ref = libraryId(query.get('asset'));
      const info = await library.assetInfo(ref);
      if (!info) return new Response(null, { status: 404 });
      return new Response(new Uint8Array(await library.assetBytes(ref)), {
        headers: {
          'content-type': info.mime,
          'cache-control': 'private, max-age=31536000, immutable',
          'x-content-type-options': 'nosniff',
          'content-security-policy': "default-src 'none'; sandbox",
        },
      });
    }
    return Response.json(
      { courses: await library.list(query.get('trash') === '1') },
      { headers: { 'cache-control': 'no-store' } },
    );
  } catch (error) {
    return failure(error);
  }
}
export async function HEAD(request: Request) {
  try {
    guard(request);
    const info = await library.assetInfo(libraryId(new URL(request.url).searchParams.get('asset')));
    return new Response(null, { status: info ? 200 : 404 });
  } catch (error) {
    return failure(error);
  }
}
export async function PUT(request: Request) {
  try {
    guard(request);
    const ref = libraryId(new URL(request.url).searchParams.get('asset'));
    if (!/^ast_[a-zA-Z0-9_-]+$/.test(ref)) throw new LibraryError('Invalid asset reference');
    if (Number(request.headers.get('content-length')) > 512 * 1024 * 1024)
      throw new LibraryError('Asset exceeds 512 MB', 413);
    const bytes = new Uint8Array(await request.arrayBuffer());
    if (bytes.length > 512 * 1024 * 1024) throw new LibraryError('Asset exceeds 512 MB', 413);
    const mime = request.headers.get('content-type') || 'application/octet-stream';
    await library.putAsset(ref, bytes, mime);
    return Response.json({ saved: true });
  } catch (error) {
    return failure(error);
  }
}
export async function POST(request: Request) {
  try {
    guard(request);
    const { operation, id, payload } = await request.json();
    let result: unknown = null;
    switch (operation) {
      case 'save':
        await library.save(payload as AppDocument);
        break;
      case 'load':
        result = await library.load(libraryId(id));
        break;
      case 'list':
        result = await library.list();
        break;
      case 'trash':
        result = await library.list(true);
        break;
      case 'delete':
        await library.remove(libraryId(id));
        break;
      case 'restore':
        await library.restore(libraryId(id));
        break;
      case 'versions':
        result = await library.versions(libraryId(id));
        break;
      case 'restoreVersion':
        await library.restoreVersion(libraryId(id), payload);
        break;
      case 'putStage':
      case 'putScene':
      case 'deleteScene':
        await library.mutate(libraryId(id), operation, payload);
        break;
      default:
        throw new LibraryError('Unknown library operation');
    }
    return Response.json({ result }, { headers: { 'cache-control': 'no-store' } });
  } catch (error) {
    return failure(error);
  }
}
