import fs from "fs";
import path from "path";
import Image from "next/image";

interface AlbumPageProps {
    params: {
        album: string;
    };
}

export async function generateStaticParams() {
    const albumsPath = path.join(process.cwd(), "public", "albums");
    const albums = fs.readdirSync(albumsPath);

    return albums.map((album) => ({ album }));
}

export default async function AlbumPage({ params }: AlbumPageProps) {
    const { album } = params;
    const albumPath = path.join(process.cwd(), "public", "albums", album);

    const filenames = fs.readdirSync(albumPath);

    const photos = filenames.map((filename) => ({
        src: `/albums/${album}/${filename}`,
    }));

  return (
    <main className="flex justify-center p-10">
        <div className="w-[900px]">
            <h1 className="text-3xl font-bold mb-6">Album: {album}</h1>
            <div className="grid grid-cols-2 md:grid-cols-2 lg:grid-cols-2 gap-4">
                {photos.map((photo) => (
                <Image
                    key={photo.src}
                    src={photo.src}
                    alt={photo.src}
                    width={400}
                    height={300}
                    className="object-cover"
                />
                ))}
            </div>
        </div>
    </main>
  );
}