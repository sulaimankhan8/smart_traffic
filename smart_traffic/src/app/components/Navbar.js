"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

export default function Navbar() {
  const router = useRouter();
  const [isOpen, setIsOpen] = useState(false);

  return (
    <nav className="bg-blue-700 text-white p-4 shadow-lg">
      <div className="container mx-auto flex justify-between items-center">
        {/* Logo */}
        <Link href="/" className="text-2xl font-bold">
          🚦 Smart Traffic
        </Link>

        {/* Mobile Menu Button */}
        <button
          className="lg:hidden text-white focus:outline-none"
          onClick={() => setIsOpen(!isOpen)}
        >
          ☰
        </button>

        {/* Menu Items */}
        <ul className={`lg:flex space-x-6 ${isOpen ? "block" : "hidden"} lg:block`}>
          <li><Link href="/" className="hover:text-gray-300">Home</Link></li>
          <li><Link href="/accident" className="hover:text-gray-300">Accident Detection</Link></li>
          <li><Link href="/DCRNN" className="hover:text-gray-300">DCRNN</Link></li>
          <li><Link href="/helmet" className="hover:text-gray-300">Helmet Detection</Link></li>
          <li><Link href="/STGCN" className="hover:text-gray-300">STGCN</Link></li>
          <li><Link href="/YOLO_TRAFFIC_LIGHT" className="hover:text-gray-300">Traffic Light Detection</Link></li>
        </ul>
      </div>
    </nav>
  );
}
