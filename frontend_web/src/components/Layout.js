import React, { useState } from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { Bars3Icon, XMarkIcon } from '@heroicons/react/24/outline';

const navigation = [
  { name: 'Dashboard', href: '/' },
  { name: 'Transcripts', href: '/transcripts' },
  { name: 'JSON Preview', href: '/json' },
];

function classNames(...classes) {
  return classes.filter(Boolean).join(' ');
}

export default function Layout() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const location = useLocation();

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="bg-primary-600 pb-32">
        <nav className="border-b border-primary-500 border-opacity-25 bg-primary-600">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div className="flex h-16 items-center justify-between">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <h1 className="text-white font-bold text-xl">Akhi Data Builder</h1>
                </div>
                <div className="hidden md:block">
                  <div className="ml-10 flex items-baseline space-x-4">
                    {navigation.map((item) => {
                      const current = location.pathname === item.href || 
                        (item.href !== '/' && location.pathname.startsWith(item.href));
                      return (
                        <Link
                          key={item.name}
                          to={item.href}
                          className={classNames(
                            current
                              ? 'bg-primary-700 text-white'
                              : 'text-white hover:bg-primary-500 hover:bg-opacity-75',
                            'rounded-md px-3 py-2 text-sm font-medium'
                          )}
                          aria-current={current ? 'page' : undefined}
                        >
                          {item.name}
                        </Link>
                      );
                    })}
                  </div>
                </div>
              </div>
              <div className="-mr-2 flex md:hidden">
                <button
                  type="button"
                  className="relative inline-flex items-center justify-center rounded-md bg-primary-600 p-2 text-primary-200 hover:bg-primary-500 hover:bg-opacity-75 hover:text-white focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-primary-600"
                  onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                >
                  <span className="absolute -inset-0.5" />
                  <span className="sr-only">Open main menu</span>
                  {mobileMenuOpen ? (
                    <XMarkIcon className="block h-6 w-6" aria-hidden="true" />
                  ) : (
                    <Bars3Icon className="block h-6 w-6" aria-hidden="true" />
                  )}
                </button>
              </div>
            </div>
          </div>

          <div className={classNames(mobileMenuOpen ? 'block' : 'hidden', 'md:hidden')}>
            <div className="space-y-1 px-2 pb-3 pt-2 sm:px-3">
              {navigation.map((item) => {
                const current = location.pathname === item.href || 
                  (item.href !== '/' && location.pathname.startsWith(item.href));
                return (
                  <Link
                    key={item.name}
                    to={item.href}
                    className={classNames(
                      current
                        ? 'bg-primary-700 text-white'
                        : 'text-white hover:bg-primary-500 hover:bg-opacity-75',
                      'block rounded-md px-3 py-2 text-base font-medium'
                    )}
                    aria-current={current ? 'page' : undefined}
                    onClick={() => setMobileMenuOpen(false)}
                  >
                    {item.name}
                  </Link>
                );
              })}
            </div>
          </div>
        </nav>
        <header className="py-10">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <h1 className="text-3xl font-bold tracking-tight text-white">
              {navigation.find(item => 
                location.pathname === item.href || 
                (item.href !== '/' && location.pathname.startsWith(item.href))
              )?.name || 'Dashboard'}
            </h1>
          </div>
        </header>
      </div>

      <main className="-mt-32">
        <div className="mx-auto max-w-7xl px-4 pb-12 sm:px-6 lg:px-8">
          <div className="rounded-lg bg-white px-5 py-6 shadow sm:px-6">
            <Outlet />
          </div>
        </div>
      </main>
    </div>
  );
}